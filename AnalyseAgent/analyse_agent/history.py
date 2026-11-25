"""Append-only text history in the private Quant-history repo (git stores only the new lines)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

CSV_COLUMNS = [
    "date", "ticker", "model", "rating", "confidence", "trend", "horizon_days",
    "close", "house_signal", "agrees_with_house", "support", "resistance", "stop_loss",
    "in_position", "report_path",
]


def append_verdicts(history_dir: Path, rows: list[dict], year: str) -> Path:
    path = history_dir / "history" / f"verdicts_{year}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)
    return path


def append_raw(history_dir: Path, records: list[dict], year: str) -> Path:
    """Full model outputs (incl. errors and token usage) — one JSON object per line."""
    path = history_dir / "history" / f"raw_{year}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return path
