"""Output contract for the daily dashboard — shared by all models, validated after every call."""
from __future__ import annotations

RATINGS = ["BUY", "ACCUMULATE", "HOLD", "REDUCE", "SELL"]
TRENDS = ["up", "down", "sideways"]

_LANG_BLOCK = {
    "type": "object",
    "additionalProperties": False,
    "required": ["headline", "summary", "bull_points", "bear_points", "risks",
                 "catalysts", "house_view_comment", "action_plan"],
    "properties": {
        "headline": {"type": "string"},
        "summary": {"type": "string"},
        "bull_points": {"type": "array", "items": {"type": "string"}},
        "bear_points": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "catalysts": {"type": "array", "items": {"type": "string"}},
        "house_view_comment": {"type": "string"},
        "action_plan": {"type": "string"},
    },
}

DASHBOARD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rating", "confidence", "horizon_days", "trend",
                 "agrees_with_house_signal", "levels", "en", "zh"],
    "properties": {
        "rating": {"type": "string", "enum": RATINGS},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "horizon_days": {"type": "integer", "minimum": 1, "maximum": 120},
        "trend": {"type": "string", "enum": TRENDS},
        "agrees_with_house_signal": {"type": "boolean"},
        "levels": {
            "type": "object",
            "additionalProperties": False,
            "required": ["support", "resistance", "stop_loss"],
            "properties": {
                "support": {"type": "number"},
                "resistance": {"type": "number"},
                "stop_loss": {"type": "number"},
            },
        },
        "en": _LANG_BLOCK,
        "zh": _LANG_BLOCK,
    },
}


def validate(v: dict) -> list[str]:
    """Light validation (no jsonschema dependency). Returns a list of problems; empty = OK."""
    errs = []
    if not isinstance(v, dict):
        return ["not a JSON object"]
    for k in DASHBOARD_SCHEMA["required"]:
        if k not in v:
            errs.append(f"missing {k}")
    if v.get("rating") not in RATINGS:
        errs.append(f"bad rating {v.get('rating')!r}")
    if v.get("trend") not in TRENDS:
        errs.append(f"bad trend {v.get('trend')!r}")
    try:
        c = int(v.get("confidence"))
        if not 0 <= c <= 100:
            errs.append("confidence out of range")
    except (TypeError, ValueError):
        errs.append("confidence not int")
    for lang in ("en", "zh"):
        blk = v.get(lang)
        if not isinstance(blk, dict):
            errs.append(f"missing {lang} block")
            continue
        for k in _LANG_BLOCK["required"]:
            if k not in blk:
                errs.append(f"{lang}.{k} missing")
    return errs
