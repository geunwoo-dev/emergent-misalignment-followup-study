from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from eval.model_utils import load_model
from runtime import maybe_mark_step, move_batch_to_device


def load_jsonl(path):
    with open(path, "r") as handle:
        return [json.loads(line) for line in handle.readlines() if line.strip()]


def save_jsonl(data, path):
    with open(path, "w") as handle:
        for row in data:
            handle.write(json.dumps(row) + "\n")


def cos_sim(a, b):
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1))


def a_proj_b(a, b):
    return (a * b).sum(dim=-1) / b.norm(dim=-1)


def main(file_path, vector_path_list=None, layer_list=None, projection_type="proj", model_name="Qwen/Qwen2.5-7B-Instruct", overwrite=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vector_path_list = vector_path_list or []
    layer_list = layer_list or []
    if not isinstance(vector_path_list, list):
        vector_path_list = [vector_path_list]
    if not isinstance(layer_list, list):
        layer_list = [layer_list]

    metric_model_name = os.path.basename(model_name) + "_"
    vector_dict = {}
    layer_dict = {}
    for vector_path in vector_path_list:
        vector = torch.load(vector_path, weights_only=False)
        for layer in layer_list:
            vector_name = os.path.basename(vector_path).split(".")[0]
            metric_name = f"{metric_model_name}{vector_name}_{projection_type}_layer{layer}"
            vector_dict[metric_name] = vector[layer]
            layer_dict[metric_name] = layer

    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path)
        prompts = [row["prompt"] for _, row in data.iterrows()]
        answers = [row["answer"] for _, row in data.iterrows()]
        for metric_name in list(vector_dict.keys()):
            if metric_name in data.columns and not overwrite:
                vector_dict.pop(metric_name)
    else:
        data = load_jsonl(file_path)
        prompts = [
            tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
            for row in data
        ]
        answers = [row["messages"][-1]["content"] for row in data]
        for metric_name in list(vector_dict.keys()):
            if metric_name in data[0] and not overwrite:
                vector_dict.pop(metric_name)

    if not vector_dict:
        print("No metrics to calculate, exiting...")
        return

    model, _ = load_model(model_name)
    projections = {metric_name: [] for metric_name in vector_dict.keys()}
    for prompt, answer in tqdm(zip(prompts, answers), total=len(prompts), desc="Calculating"):
        inputs = tokenizer(prompt + answer, return_tensors="pt", add_special_tokens=False)
        inputs = move_batch_to_device(inputs, model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for metric_name, vector in vector_dict.items():
            layer = layer_dict[metric_name]
            response_avg = outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu()
            last_prompt = outputs.hidden_states[layer][:, prompt_len - 1, :].detach().cpu()
            if projection_type == "proj":
                projection = a_proj_b(response_avg, vector).item()
            elif projection_type == "prompt_last_proj":
                projection = a_proj_b(last_prompt, vector).item()
            else:
                projection = cos_sim(response_avg, vector).item()
            projections[metric_name].append(projection)
        maybe_mark_step()

    if file_path.endswith(".csv"):
        for metric_name, values in projections.items():
            data[metric_name] = values
        data.to_csv(file_path, index=False)
    else:
        for index, row in enumerate(data):
            for metric_name, values in projections.items():
                row[metric_name] = values[index]
        save_jsonl(data, file_path)

    print(f"Projection results saved to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--vector_path_list", type=str, nargs="+", default=[])
    parser.add_argument("--layer_list", type=int, nargs="+", default=[])
    parser.add_argument("--projection_type", type=str, default="proj", choices=["proj", "prompt_last_proj", "cos_sim"])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.file_path, args.vector_path_list, args.layer_list, args.projection_type, args.model_name, args.overwrite)
