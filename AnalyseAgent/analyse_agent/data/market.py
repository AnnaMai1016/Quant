"""Market data from yfinance (free, no key). Every function degrades gracefully: on failure it
returns empty data and the fact sheet records what was missing instead of crashing the run."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

INFO_FIELDS = [
    "shortName", "sector", "industry", "marketCap", "beta",
    "trailingPE", "forwardPE", "priceToBook", "pegRatio",
    "profitMargins", "operatingMargins", "revenueGrowth", "earningsGrowth",
    "returnOnEquity", "debtToEquity", "freeCashflow",
    "dividendYield", "shortPercentOfFloat",
    "recommendationKey", "numberOfAnalystOpinions", "targetMeanPrice", "targetLowPrice", "targetHighPrice",
]


def daily_bars(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        return df.dropna(subset=["Close"])
    except Exception as e:  # network / delisted / rate limit
        log.warning("daily_bars(%s) failed: %s", ticker, e)
        return pd.DataFrame()


def intraday_bars(ticker: str, interval: str = "30m", period: str = "30d") -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        return df.dropna(subset=["Close"])
    except Exception as e:
        log.warning("intraday_bars(%s) failed: %s", ticker, e)
        return pd.DataFrame()


def fundamentals(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        log.warning("fundamentals(%s) failed: %s", ticker, e)
        return {}
    return {k: info[k] for k in INFO_FIELDS if info.get(k) is not None}


def next_earnings(ticker: str) -> str | None:
    """'YYYY-MM-DD (upcoming)' or '(already reported)' — yfinance sometimes still lists the last one."""
    try:
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            d = pd.Timestamp(dates[0]).date()
            tag = "upcoming" if d >= datetime.now(timezone.utc).date() else "already reported"
            return f"{d} ({tag})"
    except Exception as e:
        log.warning("next_earnings(%s) failed: %s", ticker, e)
    return None


def news(ticker: str, limit: int = 8) -> list[dict]:
    """Recent headlines. Handles both the old flat and the newer nested ('content') yfinance format."""
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception as e:
        log.warning("news(%s) failed: %s", ticker, e)
        return []
    out = []
    for item in raw[:limit]:
        c = item.get("content", item)
        title = c.get("title")
        if not title:
            continue
        ts = c.get("pubDate") or c.get("providerPublishTime")
        if isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        provider = c.get("provider")
        url = c.get("canonicalUrl") or c.get("link")
        out.append({
            "title": title,
            "time": str(ts or "")[:16].replace("T", " "),
            "source": provider.get("displayName") if isinstance(provider, dict) else (c.get("publisher") or ""),
            "summary": (c.get("summary") or "")[:300],
            "url": url.get("url") if isinstance(url, dict) else (url or ""),
        })
    return out


def market_snapshot(symbols: dict[str, str]) -> list[dict]:
    rows = []
    for sym, label in symbols.items():
        df = daily_bars(sym, period="1mo")
        if len(df) < 2:
            continue
        last, prev = df["Close"].iloc[-1], df["Close"].iloc[-2]
        first = df["Close"].iloc[0]
        rows.append({
            "symbol": sym, "label": label,
            "date": df.index[-1].strftime("%Y-%m-%d"),
            "close": round(float(last), 2),
            "chg_1d_pct": round(float(last / prev - 1) * 100, 2),
            "chg_1m_pct": round(float(last / first - 1) * 100, 2),
        })
    return rows
