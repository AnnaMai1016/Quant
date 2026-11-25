"""Builds the compact per-ticker fact sheet that every LLM sees (identical input for fair comparison)."""
from __future__ import annotations

from .data import market
from .features import house_signal, technicals


def build(ticker: str, settings: dict, position: dict | None) -> dict:
    hs = settings["house_signal"]
    daily = market.daily_bars(ticker, "1y")
    intraday = market.intraday_bars(ticker, hs["interval"], hs["period"])

    sheet = {
        "ticker": ticker,
        "technicals": technicals(daily),
        "house_signal": {
            "value": house_signal(intraday, hs["win"], hs["lb"], hs["thr"]),
            "explanation": (
                "Owner's own trend-following rule (median normalized slope of 4 linear fits on "
                f"{hs['interval']} bars, win={hs['win']}, lb={hs['lb']}, thr={hs['thr']}). "
                "uptrend => strategy would hold/buy; downtrend => exit; fluctuation => stay flat."
            ),
        },
        "fundamentals": market.fundamentals(ticker),
        "next_earnings": market.next_earnings(ticker),
        "news": market.news(ticker, settings.get("news_per_ticker", 8)),
        "position": None,
    }
    if position:
        close = sheet["technicals"].get("close")
        p = {k: position.get(k) for k in ("shares", "avg_cost", "note") if position.get(k) is not None}
        if close and position.get("avg_cost"):
            p["unrealized_pct"] = round((close / float(position["avg_cost"]) - 1) * 100, 2)
        sheet["position"] = p

    sheet["data_gaps"] = [k for k in ("technicals", "fundamentals", "news") if not sheet[k]]
    return sheet
