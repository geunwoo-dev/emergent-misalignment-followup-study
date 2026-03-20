from __future__ import annotations

import os
import re
from pathlib import Path

from peft import PeftConfig, PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime import get_device, get_dtype


_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def _pick_latest_checkpoint(model_path: str) -> str:
    checkpoints = []
    for child in Path(model_path).iterdir():
        match = _CHECKPOINT_RE.fullmatch(child.name)
        if match and child.is_dir():
            checkpoints.append((int(match.group(1)), child))
    if not checkpoints:
        return model_path
    return str(max(checkpoints, key=lambda item: item[0])[1])


def _is_lora(path: str) -> bool:
    return Path(path, "adapter_config.json").exists()


def _load_tokenizer(path_or_id: str):
    tokenizer = AutoTokenizer.from_pretrained(path_or_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def _move_to_runtime_device(model):
    model.to(get_device(prefer_tpu=False))
    model.eval()
    return model


def load_model(model_path: str, dtype: torch.dtype | None = None):
    dtype = dtype or get_dtype()
    if not os.path.exists(model_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
        )
        return _move_to_runtime_device(model), _load_tokenizer(model_path)

    resolved = _pick_latest_checkpoint(model_path)
    print(f"loading {resolved}")
    if _is_lora(resolved):
        config = PeftConfig.from_pretrained(resolved)
        base = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=dtype,
        )
        model = PeftModel.from_pretrained(base, resolved).merge_and_unload()
        tokenizer = _load_tokenizer(config.base_model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved,
            torch_dtype=dtype,
        )
        tokenizer = _load_tokenizer(resolved)
    return _move_to_runtime_device(model), tokenizer
