from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from itertools import islice

import pandas as pd
import torch
from tqdm import tqdm, trange

from activation_steer import ActivationSteerer
from config import setup_credentials
from eval.model_utils import load_model
from eval.prompts import Prompts
from judge import OpenAiJudge
from runtime import maybe_mark_step, move_batch_to_device, set_seed


logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

setup_credentials(require_openai=False, require_hf=False, require_wandb=False)


def chunked(iterable, size):
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


def render_prompts(tokenizer, conversations):
    return [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in conversations
    ]


def generate_batched(
    model,
    tokenizer,
    conversations,
    *,
    bs: int = 8,
    top_p: float = 1.0,
    max_tokens: int = 1000,
    temperature: float = 1.0,
    min_tokens: int = 1,
    seed: int | None = None,
    vector: torch.Tensor | None = None,
    layer: int | None = None,
    coef: float = 0.0,
    steering_type: str = "response",
):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = render_prompts(tokenizer, conversations)
    answers = []
    use_sampling = temperature > 0

    for batch_index, start in enumerate(trange(0, len(prompts), bs)):
        if seed is not None:
            set_seed(seed + batch_index)
        batch = prompts[start : start + bs]
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        tokenized = move_batch_to_device(tokenized, model.device)
        input_width = tokenized["input_ids"].shape[1]
        generate_kwargs = dict(
            **tokenized,
            do_sample=use_sampling,
            temperature=temperature if use_sampling else None,
            top_p=top_p,
            max_new_tokens=max_tokens,
            min_new_tokens=min_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if not use_sampling:
            generate_kwargs.pop("temperature")

        with torch.no_grad():
            if coef != 0 and vector is not None and layer is not None:
                with ActivationSteerer(
                    model,
                    vector,
                    coeff=coef,
                    layer_idx=layer - 1,
                    positions=steering_type,
                ):
                    output = model.generate(**generate_kwargs)
            else:
                output = model.generate(**generate_kwargs)

        batch_answers = [
            tokenizer.decode(row[input_width:], skip_special_tokens=True).strip()
            for row in output
        ]
        answers.extend(batch_answers)
        maybe_mark_step()

    return prompts, answers


def load_jsonl(path):
    with open(path, "r") as handle:
        return [json.loads(line) for line in handle.readlines() if line.strip()]


class Question:
    def __init__(
        self,
        id: str,
        paraphrases: list[str],
        judge_prompts: dict,
        temperature: float = 1.0,
        system: str | None = None,
        judge: str | None = "gpt-4o",
        judge_eval_type: str = "0_100",
        **ignored_extra_args,
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        if judge:
            self.judges = {
                metric: OpenAiJudge(
                    judge,
                    prompt,
                    eval_type=judge_eval_type if metric != "coherence" else "0_100",
                )
                for metric, prompt in judge_prompts.items()
            }
        else:
            self.judges = {}

    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        conversations = [[{"role": "user", "content": item}] for item in paraphrases]
        if self.system:
            conversations = [[{"role": "system", "content": self.system}] + convo for convo in conversations]
        return paraphrases, conversations

    async def eval(
        self,
        llm,
        tokenizer,
        coef,
        vector=None,
        layer=None,
        max_tokens=1000,
        n_per_question=100,
        steering_type="last",
        seed=None,
    ):
        paraphrases, conversations = self.get_input(n_per_question)
        prompts, answers = generate_batched(
            llm,
            tokenizer,
            conversations,
            temperature=self.temperature,
            max_tokens=max_tokens,
            seed=seed,
            vector=vector,
            layer=layer,
            coef=coef,
            steering_type=steering_type,
        )
        frame = pd.DataFrame(
            [
                {
                    "question": question,
                    "prompt": prompt,
                    "answer": answer,
                    "question_id": self.id,
                }
                for question, answer, prompt in zip(paraphrases, answers, prompts)
            ]
        )
        for score, judge in self.judges.items():
            scores = await asyncio.gather(
                *[judge(question=question, answer=answer) for question, answer in zip(paraphrases, answers)]
            )
            frame[score] = scores
        return frame


def a_or_an(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def load_persona_questions(
    trait,
    temperature=1,
    persona_instructions_type=None,
    assistant_name=None,
    judge_model="gpt-4.1-mini-2025-04-14",
    eval_type="0_100",
    version="eval",
):
    trait_data = json.load(open(f"data_generation/trait_data_{version}/{trait}.json", "r"))
    judge_prompts = {}
    if judge_model:
        judge_prompts[trait] = trait_data["eval_prompt"]
        judge_prompts["coherence"] = Prompts[f"coherence_{eval_type}"]

    questions = []
    for i, question in enumerate(trait_data["questions"]):
        if persona_instructions_type is not None:
            persona_instructions = [item[persona_instructions_type] for item in trait_data["instruction"]]
            for k, instruction in enumerate(persona_instructions):
                current_name = assistant_name
                if current_name is None:
                    current_name = trait if persona_instructions_type == "pos" else "helpful"
                system = f"You are {a_or_an(current_name)} {current_name} assistant. {instruction}"
                questions.append(
                    Question(
                        paraphrases=[question],
                        id=f"{trait}_{i}_{persona_instructions_type}_{k}",
                        judge_prompts=judge_prompts,
                        judge=judge_model,
                        temperature=temperature,
                        system=system,
                        judge_eval_type=eval_type,
                    )
                )
        else:
            questions.append(
                Question(
                    paraphrases=[question],
                    id=f"{trait}_{i}",
                    judge_prompts=judge_prompts,
                    judge=judge_model,
                    temperature=temperature,
                    judge_eval_type=eval_type,
                )
            )
    return questions


async def eval_batched(
    questions,
    llm,
    tokenizer,
    coef,
    vector=None,
    layer=None,
    n_per_question=100,
    max_concurrent_judges=100,
    max_tokens=1000,
    steering_type="last",
    seed=None,
):
    all_paraphrases = []
    all_conversations = []
    question_indices = []
    for index, question in enumerate(questions):
        paraphrases, conversations = question.get_input(n_per_question)
        all_paraphrases.extend(paraphrases)
        all_conversations.extend(conversations)
        question_indices.extend([index] * len(paraphrases))

    prompts, answers = generate_batched(
        llm,
        tokenizer,
        all_conversations,
        temperature=questions[0].temperature,
        max_tokens=max_tokens,
        seed=seed,
        vector=vector,
        layer=layer,
        coef=coef,
        steering_type=steering_type,
    )

    question_dfs = []
    all_judge_tasks = []
    all_judge_indices = []
    for index, question in enumerate(questions):
        indices = [row_index for row_index, value in enumerate(question_indices) if value == index]
        q_paraphrases = [all_paraphrases[row_index] for row_index in indices]
        q_prompts = [prompts[row_index] for row_index in indices]
        q_answers = [answers[row_index] for row_index in indices]
        frame = pd.DataFrame(
            [
                {
                    "question": question_text,
                    "prompt": prompt,
                    "answer": answer,
                    "question_id": question.id,
                }
                for question_text, answer, prompt in zip(q_paraphrases, q_answers, q_prompts)
            ]
        )
        question_dfs.append(frame)
        for metric, judge in question.judges.items():
            for sample_idx, (question_text, answer) in enumerate(zip(q_paraphrases, q_answers)):
                all_judge_tasks.append((judge, question_text, answer))
                all_judge_indices.append((index, metric, sample_idx))

    if not all_judge_tasks:
        return question_dfs

    semaphore = asyncio.Semaphore(max_concurrent_judges)
    all_results = [None] * len(all_judge_tasks)

    async def run_with_semaphore(task_idx, judge, question_text, answer):
        async with semaphore:
            result = await judge(question=question_text, answer=answer)
            return task_idx, result

    tasks = [
        run_with_semaphore(task_idx, judge, question_text, answer)
        for task_idx, (judge, question_text, answer) in enumerate(all_judge_tasks)
    ]

    with tqdm(total=len(tasks), desc="Judge evaluations") as progress:
        for task in asyncio.as_completed(tasks):
            task_idx, result = await task
            all_results[task_idx] = result
            progress.update(1)

    for task_idx, result in enumerate(all_results):
        question_idx, metric, sample_idx = all_judge_indices[task_idx]
        question_dfs[question_idx].loc[sample_idx, metric] = result

    return question_dfs


def main(
    model,
    trait,
    output_path,
    coef=0,
    vector_path=None,
    layer=None,
    steering_type="response",
    max_tokens=1000,
    n_per_question=10,
    batch_process=True,
    max_concurrent_judges=100,
    persona_instruction_type=None,
    assistant_name=None,
    judge_model="gpt-4.1-mini-2025-04-14",
    version="extract",
    overwrite=False,
    seed=0,
    skip_judging=False,
):
    if os.path.exists(output_path) and not overwrite:
        print(f"Output path {output_path} already exists, skipping...")
        frame = pd.read_csv(output_path)
        for metric in [trait, "coherence"]:
            if metric in frame.columns:
                print(f"{metric}: {frame[metric].mean():.2f} +- {frame[metric].std():.2f}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    set_seed(seed)
    temperature = 0.0 if n_per_question == 1 else 1.0

    llm, tokenizer = load_model(model)
    vector = None
    if coef != 0:
        vector = torch.load(vector_path, weights_only=False)[layer].to(llm.device)

    actual_judge_model = None if skip_judging else judge_model
    questions = load_persona_questions(
        trait,
        temperature=temperature,
        persona_instructions_type=persona_instruction_type,
        assistant_name=assistant_name,
        judge_model=actual_judge_model,
        version=version,
    )

    if batch_process:
        outputs_list = asyncio.run(
            eval_batched(
                questions,
                llm,
                tokenizer,
                coef,
                vector,
                layer,
                n_per_question,
                max_concurrent_judges,
                max_tokens,
                steering_type=steering_type,
                seed=seed,
            )
        )
        outputs = pd.concat(outputs_list)
    else:
        outputs = []
        for question in tqdm(questions, desc=f"Processing {trait} questions"):
            outputs.append(
                asyncio.run(
                    question.eval(
                        llm,
                        tokenizer,
                        coef,
                        vector,
                        layer,
                        max_tokens,
                        n_per_question,
                        steering_type=steering_type,
                        seed=seed,
                    )
                )
            )
        outputs = pd.concat(outputs)

    outputs.to_csv(output_path, index=False)
    print(output_path)
    for metric in [trait, "coherence"]:
        if metric in outputs.columns:
            print(f"{metric}: {outputs[metric].mean():.2f} +- {outputs[metric].std():.2f}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
