from __future__ import annotations

from pathlib import Path

import yaml

PKG_ROOT = Path(__file__).resolve().parent.parent  # .../AnalyseAgent


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings() -> dict:
    return _load_yaml(PKG_ROOT / "config" / "settings.yaml")


def _private_or_example(history_dir: Path, name: str) -> tuple[dict, Path]:
    """Private config lives in Quant-history/config/<name>.yaml; fall back to the example."""
    private = history_dir / "config" / f"{name}.yaml"
    if private.exists():
        return _load_yaml(private), private
    example = PKG_ROOT / "config" / f"{name}.example.yaml"
    return _load_yaml(example), example


def load_universe(history_dir: Path) -> tuple[list[str], dict[str, dict], list[Path]]:
    """Return (tickers, positions_by_ticker, config_files_used)."""
    wl, wl_path = _private_or_example(history_dir, "watchlist")
    pos, pos_path = _private_or_example(history_dir, "positions")

    positions = {}
    for p in pos.get("positions") or []:
        t = str(p["ticker"]).upper().strip()
        positions[t] = p

    tickers: list[str] = []
    for t in list(wl.get("tickers") or []) + list(positions):
        t = str(t).upper().strip()
        if t and t not in tickers:
            tickers.append(t)
    return tickers, positions, [wl_path, pos_path]
