"""DeepSeek chat completions (JSON mode). Needs DEEPSEEK_API_KEY."""
from __future__ import annotations

import os

import requests

from . import LLMError, extract_json

URL = "https://api.deepseek.com/chat/completions"


class DeepSeek:
    def __init__(self, model: str, timeout: int):
        self.key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.key:
            raise LLMError("DEEPSEEK_API_KEY not set")
        self.model, self.timeout = model, timeout

    def complete_json(self, system: str, user: str) -> tuple[dict, dict]:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "max_tokens": 4000,
        }
        r = requests.post(URL, json=body, timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.key}"})
        if r.status_code != 200:
            raise LLMError(f"deepseek HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        text = data["choices"][0]["message"]["content"] or ""
        return extract_json(text), {"usage": data.get("usage")}
