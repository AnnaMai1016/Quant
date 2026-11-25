"""Google Gemini via the AI Studio REST API (free tier works). Needs GEMINI_API_KEY."""
from __future__ import annotations

import os

import requests

from . import LLMError, extract_json

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class Gemini:
    def __init__(self, model: str, timeout: int):
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise LLMError("GEMINI_API_KEY not set")
        self.model, self.timeout = model, timeout

    def complete_json(self, system: str, user: str) -> tuple[dict, dict]:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": 8000},
        }
        r = requests.post(URL.format(model=self.model), json=body, timeout=self.timeout,
                          headers={"x-goog-api-key": self.key})
        if r.status_code != 200:
            raise LLMError(f"gemini HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError):
            raise LLMError(f"gemini returned no content: {str(data)[:300]}")
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        return extract_json(text), {"usage": data.get("usageMetadata")}
