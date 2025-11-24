from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ..config import settings


def _extract_json_object(text: str) -> Optional[dict]:
    """Extract the last JSON object from the provided text.

    The helper tolerates code fences and trailing commentary. It scans for JSON
    object boundaries and attempts to parse the last candidate.
    """

    try:
        cleaned = text.strip()
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
        if fenced:
            cleaned = fenced[-1]

        spans: list[str] = []
        depth = 0
        start_idx: int | None = None
        for idx, ch in enumerate(cleaned):
            if ch == "{":
                if depth == 0:
                    start_idx = idx
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start_idx is not None:
                        spans.append(cleaned[start_idx : idx + 1])

        for candidate in reversed(spans):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    except Exception:
        return None

    return None


class StructuredLLMClient:
    """Lightweight async wrapper around a Hugging Face text-generation pipeline."""

    def __init__(self, hf_pipeline: Any) -> None:
        self.hf_pipeline = hf_pipeline

    def _generate(self, prompt: str) -> str:
        return self.hf_pipeline(
            prompt, num_return_sequences=1, return_full_text=False
        )[0]["generated_text"]

    async def generate_text(self, prompt: str) -> str:
        return await asyncio.to_thread(self._generate, prompt)

    async def generate_json(self, prompt: str) -> dict:
        raw = await self.generate_text(prompt)
        data = _extract_json_object(raw)
        if data is None:
            raise ValueError("LLM output did not contain valid JSON")
        return data


def create_text_generation_pipeline(model_name: str | None = None, *, max_new_tokens: int = 512) -> Any:
    model_name = model_name or settings.hf_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )


def create_structured_llm_client(model_name: str | None = None, *, max_new_tokens: int = 512) -> StructuredLLMClient:
    return StructuredLLMClient(
        create_text_generation_pipeline(model_name=model_name, max_new_tokens=max_new_tokens)
    )
