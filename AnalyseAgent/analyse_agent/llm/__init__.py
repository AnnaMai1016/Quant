from __future__ import annotations

import json
import re


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> dict:
    """Parse a JSON object from model text, tolerating ```json fences or stray prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise LLMError(f"no JSON object in model output: {text[:200]!r}")


def get_client(provider: str, model: str, timeout: int):
    if provider == "claude_code":
        from .claude_code import ClaudeCode
        return ClaudeCode(model, timeout)
    if provider == "deepseek":
        from .deepseek import DeepSeek
        return DeepSeek(model, timeout)
    if provider == "gemini":
        from .gemini import Gemini
        return Gemini(model, timeout)
    raise ValueError(f"unknown provider {provider}")
