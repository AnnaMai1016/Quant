"""Technical features and the 'house signal' (your own trend_follower.py logic)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_price_trend(close_arr, win=42, lb=65, thr=0.0005) -> str:
    """Copied from IBKR/trend_follower.py (that module imports ibapi, so it can't be imported here).
    Multi-method normalized slope -> 'uptrend' | 'fluctuation' | 'downtrend' | 'unknown'."""
    n = len(close_arr)
    if n < lb + 4:
        return "unknown"
    LAMBDA = 0.8
    xwin = np.arange(win, dtype=float)
    wexp = np.array([LAMBDA ** (win - 1 - i) for i in range(win)])

    p_win = np.asarray(close_arr[n - win:], dtype=float)
    p_lb = np.asarray(close_arr[n - lb:], dtype=float)
    pmean = np.mean(p_win)
    if pmean == 0:
        return "unknown"
    sl = [np.polyfit(xwin, p_win, 1)[0] / pmean]
    ma = np.convolve(p_lb, np.ones(win) / win, mode="valid")
    if len(ma) >= 3:
        sl.append(np.polyfit(np.arange(len(ma), dtype=float), ma, 1)[0] / np.mean(ma))
    ma6 = np.convolve(p_win, np.ones(6) / 6, mode="valid")
    if len(ma6) >= 3:
        sl.append(np.polyfit(np.arange(len(ma6), dtype=float), ma6, 1)[0] / np.mean(ma6))
    sl.append(np.polyfit(xwin, p_win, 1, w=wexp)[0] / pmean)

    med = float(np.median(sl))
    if med > thr:
        return "uptrend"
    if med < -thr:
        return "downtrend"
    return "fluctuation"


def _rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return float(100 - 100 / (1 + rs.iloc[-1])) if not np.isnan(rs.iloc[-1]) else 100.0


def _r(x, nd=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def technicals(df: pd.DataFrame) -> dict:
    """Compact indicator set from daily bars (needs ~1y for SMA200)."""
    if df.empty or len(df) < 30:
        return {}
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    last = c.iloc[-1]

    def ret(n):
        return _r((last / c.iloc[-n - 1] - 1) * 100) if len(c) > n else None

    sma = {n: c.rolling(n).mean().iloc[-1] for n in (20, 50, 200)}
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    bb_mid, bb_std = c.rolling(20).mean(), c.rolling(20).std()
    bb_up, bb_lo = bb_mid + 2 * bb_std, bb_mid - 2 * bb_std
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    hi52, lo52 = h.tail(252).max(), l.tail(252).min()
    vol20 = v.tail(21).iloc[:-1].mean()
    daily_ret = c.pct_change().dropna()

    return {
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "close": _r(last),
        "ret_1d_pct": ret(1), "ret_5d_pct": ret(5), "ret_20d_pct": ret(20), "ret_60d_pct": ret(60),
        "sma20": _r(sma[20]), "sma50": _r(sma[50]), "sma200": _r(sma[200]),
        "above_sma50": bool(last > sma[50]) if not np.isnan(sma[50]) else None,
        "above_sma200": bool(last > sma[200]) if not np.isnan(sma[200]) else None,
        "rsi14": _r(_rsi(c)),
        "macd": _r(macd.iloc[-1], 3), "macd_signal": _r(signal.iloc[-1], 3),
        "macd_hist_rising": bool((macd - signal).iloc[-1] > (macd - signal).iloc[-2]),
        "bb_upper": _r(bb_up.iloc[-1]), "bb_lower": _r(bb_lo.iloc[-1]),
        "bb_pct_b": _r((last - bb_lo.iloc[-1]) / (bb_up.iloc[-1] - bb_lo.iloc[-1]), 2),
        "atr14": _r(atr), "atr14_pct": _r(atr / last * 100),
        "high_52w": _r(hi52), "low_52w": _r(lo52),
        "pct_from_52w_high": _r((last / hi52 - 1) * 100),
        "volume_vs_20d_avg": _r(v.iloc[-1] / vol20) if vol20 else None,
        "realized_vol_20d_ann_pct": _r(daily_ret.tail(20).std() * np.sqrt(252) * 100),
        "recent_swing_low_20d": _r(l.tail(20).min()), "recent_swing_high_20d": _r(h.tail(20).max()),
    }


def house_signal(intraday: pd.DataFrame, win: int, lb: int, thr: float) -> str:
    if intraday.empty:
        return "unknown"
    return compute_price_trend(intraday["Close"].to_numpy(), win=win, lb=lb, thr=thr)
