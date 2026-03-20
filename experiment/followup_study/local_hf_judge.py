from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime import get_device, get_dtype, maybe_mark_step, move_batch_to_device


class LocalHfJudge:
    def __init__(self, model_id: str, prompt_template: str, max_new_tokens: int = 8):
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=get_dtype(),
        )
        self.model.to(get_device(prefer_tpu=False))
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

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
        prompt = self.prompt_template.format(**kwargs)
        rendered = self._render_prompt(prompt)
        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = move_batch_to_device(inputs, self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        maybe_mark_step()
        return self._parse_score(text)

    async def __call__(self, **kwargs):
        return self.judge_sync(**kwargs)
