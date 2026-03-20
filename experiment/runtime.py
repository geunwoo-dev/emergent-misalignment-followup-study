from __future__ import annotations

import os
import random

import numpy as np
import torch

try:
    import torch_xla.core.xla_model as xm
except Exception:  # pragma: no cover - optional dependency
    xm = None


def xla_available() -> bool:
    return xm is not None and os.environ.get("PJRT_DEVICE", "").upper() == "TPU"


def get_device(prefer_tpu: bool = True) -> torch.device:
    if prefer_tpu and xla_available():
        return xm.xla_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_dtype() -> torch.dtype:
    if xla_available():
        return torch.bfloat16
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def use_bfloat16() -> bool:
    return get_dtype() == torch.bfloat16


def use_float16() -> bool:
    return get_dtype() == torch.float16


def maybe_mark_step() -> None:
    if xla_available():
        xm.mark_step()


def is_main_process() -> bool:
    if xla_available():
        return xm.is_master_ordinal(local=False)
    return True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }
