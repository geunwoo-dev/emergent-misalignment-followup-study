from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm

from eval.model_utils import load_model
from runtime import maybe_mark_step, move_batch_to_device


def get_hidden_p_and_r(model, tokenizer, prompts, responses, layer_list=None):
    max_layer = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(max_layer + 1))
    prompt_avg = [[] for _ in range(max_layer + 1)]
    response_avg = [[] for _ in range(max_layer + 1)]
    prompt_last = [[] for _ in range(max_layer + 1)]
    texts = [prompt + response for prompt, response in zip(prompts, responses)]

    for text, prompt in tqdm(zip(texts, prompts), total=len(texts)):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        inputs = move_batch_to_device(inputs, model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for layer in layer_list:
            prompt_avg[layer].append(outputs.hidden_states[layer][:, :prompt_len, :].mean(dim=1).detach().cpu())
            response_avg[layer].append(outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu())
            prompt_last[layer].append(outputs.hidden_states[layer][:, prompt_len - 1, :].detach().cpu())
        maybe_mark_step()

    for layer in layer_list:
        prompt_avg[layer] = torch.cat(prompt_avg[layer], dim=0)
        prompt_last[layer] = torch.cat(prompt_last[layer], dim=0)
        response_avg[layer] = torch.cat(response_avg[layer], dim=0)
    return prompt_avg, prompt_last, response_avg


def get_persona_effective(pos_path, neg_path, trait, threshold=50):
    persona_pos = pd.read_csv(pos_path)
    persona_neg = pd.read_csv(neg_path)
    mask = (
        (persona_pos[trait] >= threshold)
        & (persona_neg[trait] < 100 - threshold)
        & (persona_pos["coherence"] >= 50)
        & (persona_neg["coherence"] >= 50)
    )

    persona_pos_effective = persona_pos[mask]
    persona_neg_effective = persona_neg[mask]
    return (
        persona_pos_effective,
        persona_neg_effective,
        persona_pos_effective["prompt"].tolist(),
        persona_neg_effective["prompt"].tolist(),
        persona_pos_effective["answer"].tolist(),
        persona_neg_effective["answer"].tolist(),
    )


def save_persona_vector(model_name, pos_path, neg_path, trait, save_dir, threshold=50):
    model, tokenizer = load_model(model_name)
    (
        _,
        _,
        pos_prompts,
        neg_prompts,
        pos_responses,
        neg_responses,
    ) = get_persona_effective(pos_path, neg_path, trait, threshold)

    prompt_avg = {}
    prompt_last = {}
    response_avg = {}

    prompt_avg["pos"], prompt_last["pos"], response_avg["pos"] = get_hidden_p_and_r(
        model,
        tokenizer,
        pos_prompts,
        pos_responses,
    )
    prompt_avg["neg"], prompt_last["neg"], response_avg["neg"] = get_hidden_p_and_r(
        model,
        tokenizer,
        neg_prompts,
        neg_responses,
    )

    prompt_avg_diff = torch.stack(
        [prompt_avg["pos"][layer].mean(0).float() - prompt_avg["neg"][layer].mean(0).float() for layer in range(len(prompt_avg["pos"]))],
        dim=0,
    )
    response_avg_diff = torch.stack(
        [response_avg["pos"][layer].mean(0).float() - response_avg["neg"][layer].mean(0).float() for layer in range(len(response_avg["pos"]))],
        dim=0,
    )
    prompt_last_diff = torch.stack(
        [prompt_last["pos"][layer].mean(0).float() - prompt_last["neg"][layer].mean(0).float() for layer in range(len(prompt_last["pos"]))],
        dim=0,
    )

    os.makedirs(save_dir, exist_ok=True)
    torch.save(prompt_avg_diff, f"{save_dir}/{trait}_prompt_avg_diff.pt")
    torch.save(response_avg_diff, f"{save_dir}/{trait}_response_avg_diff.pt")
    torch.save(prompt_last_diff, f"{save_dir}/{trait}_prompt_last_diff.pt")
    print(f"Trait vectors saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--pos_path", type=str, required=True)
    parser.add_argument("--neg_path", type=str, required=True)
    parser.add_argument("--trait", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--threshold", type=int, default=50)
    args = parser.parse_args()
    save_persona_vector(
        args.model_name,
        args.pos_path,
        args.neg_path,
        args.trait,
        args.save_dir,
        args.threshold,
    )
