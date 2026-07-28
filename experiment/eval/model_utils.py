from __future__ import annotations

import os
import re
from pathlib import Path

from peft import PeftConfig, PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    if hasattr(model, "hf_device_map"):
        model.eval()
        return model
    model.to(get_device(prefer_tpu=False))
    model.eval()
    return model


def _load_kwargs(dtype: torch.dtype) -> dict:
    load_in_4bit = os.environ.get("EM_EVAL_LOAD_IN_4BIT", "0").lower() in {"1", "true", "yes"}
    if not load_in_4bit or not torch.cuda.is_available():
        return {"torch_dtype": dtype}
    return {
        "torch_dtype": dtype,
        "device_map": "auto",
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
    }


def load_model(model_path: str, dtype: torch.dtype | None = None):
    dtype = dtype or get_dtype()
    load_kwargs = _load_kwargs(dtype)
    if not os.path.exists(model_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs,
        )
        return _move_to_runtime_device(model), _load_tokenizer(model_path)

    resolved = _pick_latest_checkpoint(model_path)
    print(f"loading {resolved}")
    if _is_lora(resolved):
        config = PeftConfig.from_pretrained(resolved)
        base = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            **load_kwargs,
        )
        model = PeftModel.from_pretrained(base, resolved)
        should_merge = os.environ.get("EM_EVAL_MERGE_LORA", "0").lower() in {"1", "true", "yes"}
        if should_merge:
            if load_kwargs.get("quantization_config") is not None:
                raise ValueError("EM_EVAL_MERGE_LORA=1 is incompatible with 4-bit evaluation")
            model = model.merge_and_unload()
        tokenizer = _load_tokenizer(config.base_model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved,
            **load_kwargs,
        )
        tokenizer = _load_tokenizer(resolved)
    return _move_to_runtime_device(model), tokenizer
