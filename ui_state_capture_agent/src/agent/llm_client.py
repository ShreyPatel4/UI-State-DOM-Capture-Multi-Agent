from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Optional

from openai import OpenAI

from ..config import settings


class OpenAIChatPipeline:
    def __init__(self, model: str, api_key: str, base_url: str | None, max_new_tokens: int):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt, max_new_tokens: int | None = None, **_):
        max_tokens = max_new_tokens or self.max_new_tokens
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=max_tokens,
        )
        return [{"generated_text": resp.choices[0].message.content}]

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


class PolicyLLMClient:
    def __init__(self, hf_pipeline: Any) -> None:
        self.pipeline = hf_pipeline

    def generate_text(self, prompt: str) -> str:
        try:
            out = self.pipeline(
                prompt,
                max_new_tokens=128,
                do_sample=False,
                return_full_text=False,
            )
        except Exception:
            raise

        if isinstance(out, list) and out:
            item = out[0]
            if isinstance(item, dict) and "generated_text" in item:
                return item["generated_text"].strip()
            if isinstance(item, str):
                return item.strip()

        return str(out).strip()


def create_text_generation_pipeline(model_name: str | None = None, *, max_new_tokens: int = 512):
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when llm_provider=openai")
        return OpenAIChatPipeline(
            model=model_name or settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(f"Unsupported llm_provider: {settings.llm_provider}")

def create_structured_llm_client(model_name: str | None = None, *, max_new_tokens: int = 512) -> StructuredLLMClient:
    return StructuredLLMClient(
        create_text_generation_pipeline(model_name=model_name, max_new_tokens=max_new_tokens)
    )
