"""Claude via headless Claude Code (`claude -p`). Auth: CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`
(your Claude Pro subscription), or whatever login the local CLI already has.
All tools are disabled: the model only reads the fact sheet we pass in and returns JSON."""
from __future__ import annotations

import json
import os
import shutil
import subprocess

from . import LLMError, extract_json
from ..schema import DASHBOARD_SCHEMA


def _find_binary() -> str:
    exe = os.environ.get("CLAUDE_BIN") or shutil.which("claude")
    if not exe:
        raise LLMError("claude CLI not found (install @anthropic-ai/claude-code or set CLAUDE_BIN)")
    return exe


class ClaudeCode:
    def __init__(self, model: str, timeout: int):
        self.model = model
        self.timeout = timeout
        self.exe = _find_binary()

    def complete_json(self, system: str, user: str) -> tuple[dict, dict]:
        cmd = [
            self.exe, "-p",
            "--model", self.model,
            "--output-format", "json",
            "--system-prompt", system,
            "--json-schema", json.dumps(DASHBOARD_SCHEMA),
            "--tools", "",
            "--strict-mcp-config",
            "--no-session-persistence",
        ]
        try:
            proc = subprocess.run(cmd, input=user, capture_output=True, text=True,
                                  encoding="utf-8", timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise LLMError(f"claude timed out after {self.timeout}s")
        if proc.returncode != 0 and not proc.stdout.strip():
            raise LLMError(f"claude exit {proc.returncode}: {proc.stderr.strip()[:300]}")
        try:
            env = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise LLMError(f"claude returned non-JSON envelope: {proc.stdout[:200]!r}")
        if env.get("is_error"):
            raise LLMError(f"claude error: {str(env.get('result'))[:300]}")
        meta = {"cost_usd": env.get("total_cost_usd"), "usage": env.get("usage")}
        if isinstance(env.get("structured_output"), dict):
            return env["structured_output"], meta
        return extract_json(env.get("result") or ""), meta
