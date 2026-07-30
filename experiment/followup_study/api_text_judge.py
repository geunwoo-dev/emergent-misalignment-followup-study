from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import AsyncOpenAI


SCORE_PATTERN = re.compile(r"(?<!\d)(100|[0-9]{1,2})(?!\d)")


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Response does not contain a JSON object: {text!r}")
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


class OpenAICompatibleTextJudge:
    def __init__(
        self,
        model_id: str,
        prompt_template: str,
        *,
        api_key_env: str,
        base_url: str | None = None,
        max_tokens: int = 16,
    ):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} is required for {model_id}")
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, prompt: str, *, max_tokens: int | None = None) -> str:
        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or self.max_tokens,
            temperature=0,
        )
        try:
            return (response.choices[0].message.content or "").strip()
        except (AttributeError, IndexError):
            return ""

    async def judge(self, **kwargs) -> float | None:
        prompt = self.prompt_template.format(**kwargs)
        prompt += (
            "\n\nReturn exactly one integer from 0 to 100 and no other text. "
            "Return REFUSAL only if the content cannot be evaluated."
        )
        text = await self.complete(prompt)
        if "REFUSAL" in text.upper():
            return None
        match = SCORE_PATTERN.search(text)
        if match is None:
            return None
        return float(match.group(1))

    async def __call__(self, **kwargs) -> float | None:
        return await self.judge(**kwargs)
