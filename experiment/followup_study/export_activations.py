from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eval.model_utils import load_model
from runtime import maybe_mark_step, move_batch_to_device, set_seed


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_indices(length: int, max_tokens: int) -> list[int]:
    if length <= max_tokens:
        return list(range(length))
    if max_tokens <= 1:
        return [length - 1]
    step = (length - 1) / (max_tokens - 1)
    return [round(index * step) for index in range(max_tokens)]


def build_chat_text(tokenizer, messages: list[dict]) -> tuple[str, str]:
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    answer = messages[-1]["content"]
    return prompt, answer


def export_activations(
    model_name: str,
    dataset_path: Path,
    output_path: Path,
    layer: int,
    token_source: str,
    max_samples: int,
    max_tokens_per_sample: int,
    seed: int,
) -> None:
    set_seed(seed)
    rows = load_jsonl(dataset_path)
    if max_samples and len(rows) > max_samples:
        rows = rows[:max_samples]

    model, tokenizer = load_model(model_name)
    exported = []
    metadata = []

    for row_idx, row in enumerate(rows):
        prompt, answer = build_chat_text(tokenizer, row["messages"])
        full_text = prompt + answer
        inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        inputs = move_batch_to_device(inputs, model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer][0].detach().cpu().float()

        if token_source == "prompt_last":
            vectors = hidden[prompt_len - 1 : prompt_len]
            token_indices = [prompt_len - 1]
        elif token_source == "response_mean":
            vectors = hidden[prompt_len:, :].mean(dim=0, keepdim=True)
            token_indices = [-1]
        else:
            response_hidden = hidden[prompt_len:, :]
            chosen = sample_indices(len(response_hidden), max_tokens_per_sample)
            vectors = response_hidden[chosen]
            token_indices = chosen

        exported.append(vectors)
        metadata.extend(
            {
                "row_idx": row_idx,
                "token_idx": token_idx,
                "token_source": token_source,
            }
            for token_idx in token_indices
        )
        maybe_mark_step()

    activations = torch.cat(exported, dim=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "dataset_path": str(dataset_path),
            "layer": layer,
            "token_source": token_source,
            "activations": activations,
            "metadata": metadata,
        },
        output_path,
    )
    print(f"Saved {activations.shape[0]} activation vectors to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--token_source", choices=["response", "response_mean", "prompt_last"], default="response")
    parser.add_argument("--max_samples", type=int, default=512)
    parser.add_argument("--max_tokens_per_sample", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    export_activations(
        model_name=args.model,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        layer=args.layer,
        token_source=args.token_source,
        max_samples=args.max_samples,
        max_tokens_per_sample=args.max_tokens_per_sample,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
