from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from runtime import get_device, get_dtype, maybe_mark_step, move_batch_to_device


class LocalHfJudge:
    _MODEL_CACHE: dict[str, tuple[AutoModelForCausalLM, AutoTokenizer]] = {}

    def __init__(self, model_id: str, prompt_template: str, max_new_tokens: int = 8):
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = self._get_or_load_model(model_id)

    @classmethod
    def _get_or_load_model(cls, model_id: str):
        cached = cls._MODEL_CACHE.get(model_id)
        if cached is not None:
            return cached

        dtype = get_dtype()
        device = get_device(prefer_tpu=False)
        load_kwargs = {
            "torch_dtype": dtype,
        }
        if device.type == "cuda":
            load_kwargs["device_map"] = "auto"
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **load_kwargs,
        )
        if device.type != "cuda":
            model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        cls._MODEL_CACHE[model_id] = (model, tokenizer)
        return cls._MODEL_CACHE[model_id]

    def _render_prompt(self, text: str) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return text

    def _parse_score(self, text: str):
        upper = text.upper()
        if "REFUSAL" in upper:
            return None
        match = re.search(r"\b(100|[0-9]{1,2})\b", text)
        if match is None:
            return None
        score = int(match.group(1))
        if 0 <= score <= 100:
            return float(score)
        return None

    def judge_sync(self, **kwargs):
        return self.judge_batch_sync([kwargs], batch_size=1)[0]

    def judge_batch_sync(
        self,
        items: list[dict],
        batch_size: int = 32,
    ) -> list[float | None]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        rendered_prompts = [
            self._render_prompt(self.prompt_template.format(**item))
            for item in items
        ]
        scores: list[float | None] = []
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            for start in range(0, len(rendered_prompts), batch_size):
                batch = rendered_prompts[start : start + batch_size]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                inputs = move_batch_to_device(inputs, self.model.device)
                input_width = inputs["input_ids"].shape[1]
                with torch.inference_mode():
                    output = self.model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=self.max_new_tokens,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                generated = output[:, input_width:]
                texts = self.tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                )
                scores.extend(self._parse_score(text.strip()) for text in texts)
                maybe_mark_step()
        finally:
            self.tokenizer.padding_side = original_padding_side
        return scores

    async def __call__(self, **kwargs):
        return self.judge_sync(**kwargs)
