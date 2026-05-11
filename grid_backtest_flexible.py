from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from util_Grid import buy_partial, sell_partial, add_tradelog

AnchorMethod = Literal["VWAP", "OPEN15_MID"]
GridMethod = Literal["ATR_MULT", "PCT", "NOISE", "HYBRID"]
LayerMethod = Literal["FIXED", "HIST30"]


@dataclass
class AnchorConfig:
    method: AnchorMethod = "VWAP"
    open_minutes: int = 15
    update_interval_minutes: int = 5


@dataclass
class GridSizeConfig:
    method: GridMethod = "HYBRID"
    atr_mult: float = 0.25           # for ATR_MULT
    pct: float = 0.003               # for PCT, 0.3%
    noise_mult: float = 3.0          # for NOISE
    spread_mult: float = 5.0         # for NOISE / HYBRID
    tick_mult: float = 2.0           # for HYBRID
    atr_mult_hybrid: float = 0.25    # for HYBRID
    atr_window: int = 20
    adx_window: int = 14
    noise_window: int = 20
    min_grid_pct: float = 0.0005     # 5 bps lower bound
    max_grid_pct: float = 0.03       # 300 bps upper bound
    fallback_spread_bps: float = 1.0 # if no spread/bid/ask columns exist


@dataclass
class LayerConfig:
    method: LayerMethod = "FIXED"
    fixed_layers: int = 5
    hist_lookback_days: int = 30
    min_layers: int = 3
    max_layers: int = 12


@dataclass
class ResetConfig:
    minutes_same_side: int = 20
    anchor_shift_grids: float = 1.5
    adx_threshold: float = 30.0
    midday_reanchor: bool = True
    midday_time: str = "12:00"
    reset_on_adx_trend: bool = True
    reset_cooldown_minutes: int = 10


@dataclass
class RiskConfig:
    fee_rate: float = 0.0005
    stop_pct_from_anchor: float = 0.03
    max_layers_holding_bars: Optional[int] = None
    force_close_end_of_day: bool = True
    session_close_time: Optional[str] = None   # e.g. '15:55'; if None, close on session boundary
    price_field_for_execution: Literal["close", "open"] = "close"


@dataclass
class BacktestConfig:
    initial_cash: float = 100_000.0
    units_per_layer: int = 100
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    grid: GridSizeConfig = field(default_factory=GridSizeConfig)
    layers: LayerConfig = field(default_factory=LayerConfig)
    reset: ResetConfig = field(default_factory=ResetConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


# ----------------------------- Indicator helpers -----------------------------

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"])
        out = out.sort_values("datetime").set_index("datetime")
    elif isinstance(out.index, pd.DatetimeIndex):
        out = out.sort_index()
    elif {"date", "time"}.issubset(out.columns):
        out["datetime"] = pd.to_datetime(out["date"].astype(str) + " " + out["time"].astype(str))
        out = out.sort_values("datetime").set_index("datetime")
    elif "date" in out.columns:
        out["datetime"] = pd.to_datetime(out["date"])
        out = out.sort_values("datetime").set_index("datetime")
    else:
        raise ValueError("df must contain a DatetimeIndex or a datetime/date column")
    return out


def _infer_tick_size(close: pd.Series) -> float:
    diffs = close.dropna().diff().abs()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 0.01
    q = np.nanpercentile(diffs.values, 5)
    if q <= 0 or not np.isfinite(q):
        return 0.01
    return float(q)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    return _true_range(df).rolling(window, min_periods=window).mean()


def _adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return adx


def _session_vwap(df: pd.DataFrame) -> pd.Series:
    vol = df["volume"].fillna(0.0)
    pxv = (df["close"] * vol).groupby(df["session_date"]).cumsum()
    cvol = vol.groupby(df["session_date"]).cumsum().replace(0, np.nan)
    return pxv / cvol


def _open15_mid(df: pd.DataFrame, open_minutes: int) -> pd.Series:
    result = pd.Series(index=df.index, dtype=float)
    for _, g in df.groupby("session_date", sort=True):
        first_block = g.iloc[:open_minutes]
        if first_block.empty:
            continue
        mid = 0.5 * (first_block["high"].max() + first_block["low"].min())
        result.loc[g.index] = float(mid)
    return result


def _compute_spread(df: pd.DataFrame, fallback_spread_bps: float) -> pd.Series:
    if "spread" in df.columns:
        sp = pd.to_numeric(df["spread"], errors="coerce")
    elif {"ask", "bid"}.issubset(df.columns):
        sp = pd.to_numeric(df["ask"], errors="coerce") - pd.to_numeric(df["bid"], errors="coerce")
    else:
        sp = df["close"] * (fallback_spread_bps / 10_000.0)
    return sp.clip(lower=0)


def prepare_intraday_features(raw_df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    df = _ensure_datetime_index(raw_df)
    req = ["open", "high", "low", "close", "volume"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for c in req:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=req).copy()

    df["session_date"] = pd.Index(df.index.date)
    df["minute_of_session"] = df.groupby("session_date").cumcount() + 1
    df["session_time"] = pd.Index(df.index.strftime("%H:%M"))

    df["atr_abs"] = _atr(df, cfg.grid.atr_window)
    df["atr_pct"] = (df["atr_abs"] / df["close"]).clip(lower=cfg.grid.min_grid_pct)
    df["adx"] = _adx(df, cfg.grid.adx_window)
    df["spread"] = _compute_spread(df, cfg.grid.fallback_spread_bps)
    df["tick_size"] = _infer_tick_size(df["close"])

    abs_move = df["close"].diff().abs()
    df["noise_width"] = abs_move.rolling(cfg.grid.noise_window, min_periods=max(5, cfg.grid.noise_window // 2)).median()

    df["vwap"] = _session_vwap(df)
    df["open15_mid"] = _open15_mid(df, cfg.anchor.open_minutes)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()

    # grid candidates
    df["grid_atr_mult"] = (cfg.grid.atr_mult * df["atr_pct"]).clip(cfg.grid.min_grid_pct, cfg.grid.max_grid_pct)
    df["grid_pct"] = np.clip(cfg.grid.pct, cfg.grid.min_grid_pct, cfg.grid.max_grid_pct)

    noise_pct = (cfg.grid.noise_mult * df["noise_width"] / df["close"]).clip(cfg.grid.min_grid_pct, cfg.grid.max_grid_pct)
    spread_pct = (cfg.grid.spread_mult * df["spread"] / df["close"]).clip(cfg.grid.min_grid_pct, cfg.grid.max_grid_pct)
    tick_pct = (cfg.grid.tick_mult * df["tick_size"] / df["close"]).clip(cfg.grid.min_grid_pct, cfg.grid.max_grid_pct)

    df["grid_noise"] = np.maximum.reduce([noise_pct.fillna(0), spread_pct.fillna(0)])
    df["grid_hybrid"] = np.maximum.reduce([
        (cfg.grid.atr_mult_hybrid * df["atr_pct"]).fillna(0),
        spread_pct.fillna(0),
        tick_pct.fillna(0),
    ])
    df["grid_noise"] = pd.Series(df["grid_noise"], index=df.index).replace(0, np.nan).clip(cfg.grid.min_grid_pct, cfg.grid.max_grid_pct)
    df["grid_hybrid"] = pd.Series(df["grid_hybrid"], index=df.index).replace(0, np.nan).clip(cfg.grid.min_grid_pct, cfg.grid.max_grid_pct)

    return df


# ----------------------------- Strategy class -----------------------------

class FlexibleIntradayGrid:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.anchor: float = np.nan
        self.grid_pct: float = np.nan
        self.max_layers: int = cfg.layers.fixed_layers
        self.layer: int = 0
        self.next_buy: float = np.nan
        self.next_sell: float = np.nan
        self.current_session = None
        self.current_trade_seq = 0
        self.layer_sizes: Dict[int, int] = {}
        self.last_reset_time: Optional[pd.Timestamp] = None
        self.same_side_count = 0
        self.last_side = 0
        self.midday_done_for_session = False
        self.max_layer_entry_bar: Optional[int] = None

    def _grid_col(self) -> str:
        mapping = {
            "ATR_MULT": "grid_atr_mult",
            "PCT": "grid_pct",
            "NOISE": "grid_noise",
            "HYBRID": "grid_hybrid",
        }
        return mapping[self.cfg.grid.method]

    def _anchor_col(self) -> str:
        return "vwap" if self.cfg.anchor.method == "VWAP" else "open15_mid"

    def _can_reset(self, ts: pd.Timestamp) -> bool:
        if self.last_reset_time is None:
            return True
        delta_min = (ts - self.last_reset_time).total_seconds() / 60.0
        return delta_min >= self.cfg.reset.reset_cooldown_minutes

    def _resolve_layers(self, df: pd.DataFrame, i: int, anchor: float, grid_pct: float) -> int:
        if self.cfg.layers.method == "FIXED":
            return self.cfg.layers.fixed_layers

        session_date = df["session_date"].iloc[i]
        sessions = sorted(df["session_date"].unique())
        pos = sessions.index(session_date)
        hist_sessions = sessions[max(0, pos - self.cfg.layers.hist_lookback_days):pos]
        if not hist_sessions:
            return self.cfg.layers.fixed_layers

        hist_ranges = []
        for s in hist_sessions:
            g = df[df["session_date"] == s]
            if g.empty:
                continue
            # Use intraday range around session VWAP midpoint proxy
            center = float(g["vwap"].dropna().iloc[-1]) if g["vwap"].notna().any() else float(g["close"].median())
            if center <= 0:
                continue
            day_range_pct = (g["high"].max() - g["low"].min()) / center
            hist_ranges.append(day_range_pct)
        if not hist_ranges or not np.isfinite(grid_pct) or grid_pct <= 0:
            return self.cfg.layers.fixed_layers

        expected_range_pct = float(np.nanmedian(hist_ranges))
        one_side_layers = math.ceil(0.5 * expected_range_pct / grid_pct)
        return int(np.clip(one_side_layers, self.cfg.layers.min_layers, self.cfg.layers.max_layers))

    def _reset_grid(self, ts: pd.Timestamp, anchor: float, grid_pct: float, max_layers: int, trade_log: List[dict], state: dict,
                    reason: str, price_for_log: float = 0.0, keep_layer: int = 0) -> None:
        self.anchor = float(anchor)
        self.grid_pct = float(grid_pct)
        self.max_layers = int(max_layers)
        self.layer = int(keep_layer)
        self.layer_sizes = {} if keep_layer == 0 else self.layer_sizes
        self.same_side_count = 0
        self.last_side = 0
        self.last_reset_time = ts
        self.midday_done_for_session = False if reason == "NEW_SESSION" else self.midday_done_for_session
        self.max_layer_entry_bar = None
        self.next_buy = self.anchor * (1.0 - (self.layer + 1) * self.grid_pct)
        self.next_sell = self.anchor * (1.0 + max(self.layer, 1) * self.grid_pct) if self.layer > 0 else self.anchor * (1.0 + self.grid_pct)
        add_tradelog(
            trade_log,
            ttype=f"[RESET ANCHOR:{reason}]",
            layer=self.layer,
            price=price_for_log,
            size=0,
            time=ts,
            trade_time=999,
            fee=self.cfg.risk.fee_rate,
            position=state["position"],
            anchor=self.anchor,
            next_buy=self.next_buy,
            next_sell=self.next_sell,
        )

    def _force_close(self, ts: pd.Timestamp, px: float, state: dict, trade_log: List[dict], reason: str) -> None:
        if state["position"] <= 0:
            return
        sell_size = state["position"]
        sell_partial(state, price=px, size=sell_size, row_time=ts, fee=self.cfg.risk.fee_rate)
        self.current_trade_seq += 1
        add_tradelog(
            trade_log,
            ttype=f"SELL[{reason}]",
            layer=self.layer,
            price=px,
            size=sell_size,
            time=ts,
            trade_time=self.current_trade_seq,
            fee=self.cfg.risk.fee_rate,
            position=state["position"],
            anchor=self.anchor,
            next_buy=self.next_buy,
            next_sell=self.next_sell,
        )
        self.layer = 0
        self.layer_sizes = {}
        self.max_layer_entry_bar = None

    def _update_same_side_counter(self, px: float) -> None:
        if not np.isfinite(self.anchor):
            self.same_side_count = 0
            self.last_side = 0
            return
        side = 1 if px > self.anchor else (-1 if px < self.anchor else 0)
        if side != 0 and side == self.last_side:
            self.same_side_count += 1
        elif side != 0:
            self.same_side_count = 1
            self.last_side = side
        else:
            self.same_side_count = 0
            self.last_side = 0

    def _maybe_reset(self, df: pd.DataFrame, i: int, row: pd.Series, state: dict, trade_log: List[dict]) -> None:
        ts = row.name
        px = float(row["close"])
        self._update_same_side_counter(px)

        reasons = []
        if self.same_side_count >= self.cfg.reset.minutes_same_side:
            reasons.append("SAME_SIDE_TOO_LONG")

        anchor_ref = float(row[self._anchor_col()]) if pd.notna(row[self._anchor_col()]) else np.nan
        if np.isfinite(anchor_ref) and np.isfinite(self.anchor) and np.isfinite(self.grid_pct):
            shift_grids = abs(anchor_ref - self.anchor) / (self.anchor * self.grid_pct) if self.anchor * self.grid_pct > 0 else 0
            if shift_grids >= self.cfg.reset.anchor_shift_grids:
                reasons.append("ANCHOR_DRIFT")

        if self.cfg.reset.reset_on_adx_trend and pd.notna(row["adx"]) and float(row["adx"]) >= self.cfg.reset.adx_threshold:
            reasons.append("ADX_HIGH")

        if self.cfg.reset.midday_reanchor and not self.midday_done_for_session:
            hhmm = ts.strftime("%H:%M")
            if hhmm >= self.cfg.reset.midday_time:
                reasons.append("MIDDAY")
                self.midday_done_for_session = True

        if not reasons or not self._can_reset(ts):
            return

        new_anchor = anchor_ref if np.isfinite(anchor_ref) else px
        new_grid = float(row[self._grid_col()]) if pd.notna(row[self._grid_col()]) else self.grid_pct
        new_layers = self._resolve_layers(df, i, new_anchor, new_grid)
        self._reset_grid(ts, new_anchor, new_grid, new_layers, trade_log, state, "+".join(reasons), price_for_log=px, keep_layer=0)

    def _maybe_hard_stop(self, row: pd.Series, i: int, state: dict, trade_log: List[dict]) -> bool:
        if state["position"] <= 0 or not np.isfinite(self.anchor):
            return False
        ts = row.name
        px = float(row[self.cfg.risk.price_field_for_execution])
        if px <= self.anchor * (1.0 - self.cfg.risk.stop_pct_from_anchor):
            self._force_close(ts, px, state, trade_log, reason="HardStop")
            return True
        if self.cfg.risk.max_layers_holding_bars is not None and self.max_layer_entry_bar is not None:
            if i - self.max_layer_entry_bar >= self.cfg.risk.max_layers_holding_bars:
                self._force_close(ts, px, state, trade_log, reason="MaxLayerHolding")
                return True
        return False

    def _maybe_trade(self, row: pd.Series, state: dict, trade_log: List[dict], i: int) -> None:
        px = float(row[self.cfg.risk.price_field_for_execution])
        ts = row.name
        if not np.isfinite(self.anchor) or not np.isfinite(self.grid_pct):
            return

        # Update current ladder levels first.
        self.next_buy = self.anchor * (1.0 - (self.layer + 1) * self.grid_pct)
        self.next_sell = self.anchor * (1.0 + self.layer * self.grid_pct) if self.layer > 0 else self.anchor * (1.0 + self.grid_pct)

        # Buy loop: allow gap-through multiple levels in one bar.
        while self.layer < self.max_layers and px <= self.next_buy and state["cash"] > 0:
            self.current_trade_seq += 1
            self.layer += 1
            size = int(self.cfg.initial_cash * 0 + self.cfg.units_per_layer)
            self.layer_sizes[self.layer] = size
            buy_partial(state, price=px, size=size, row_time=ts, fee=self.cfg.risk.fee_rate)
            add_tradelog(
                trade_log,
                ttype="BUY",
                layer=self.layer,
                price=px,
                size=size,
                time=ts,
                trade_time=self.current_trade_seq,
                fee=self.cfg.risk.fee_rate,
                position=state["position"],
                anchor=self.anchor,
                next_buy=self.next_buy,
                next_sell=self.next_sell,
            )
            if self.layer == self.max_layers:
                self.max_layer_entry_bar = i
            self.next_buy = self.anchor * (1.0 - (self.layer + 1) * self.grid_pct)
            self.next_sell = self.anchor * (1.0 + self.layer * self.grid_pct)

        # Sell loop: unwind symmetrically.
        while self.layer > 0 and px >= self.next_sell and state["position"] > 0:
            self.current_trade_seq += 1
            size = min(self.layer_sizes.get(self.layer, self.cfg.units_per_layer), int(state["position"]))
            sell_partial(state, price=px, size=size, row_time=ts, fee=self.cfg.risk.fee_rate)
            add_tradelog(
                trade_log,
                ttype="SELL",
                layer=self.layer,
                price=px,
                size=size,
                time=ts,
                trade_time=self.current_trade_seq,
                fee=self.cfg.risk.fee_rate,
                position=state["position"],
                anchor=self.anchor,
                next_buy=self.next_buy,
                next_sell=self.next_sell,
            )
            self.layer -= 1
            self.next_buy = self.anchor * (1.0 - (self.layer + 1) * self.grid_pct)
            self.next_sell = self.anchor * (1.0 + self.layer * self.grid_pct) if self.layer > 0 else self.anchor * (1.0 + self.grid_pct)
            if self.layer == 0:
                self.layer_sizes = {}
                self.max_layer_entry_bar = None

    def initialize_session(self, df: pd.DataFrame, i: int, row: pd.Series, state: dict, trade_log: List[dict]) -> None:
        ts = row.name
        self.current_session = row["session_date"]
        self.current_trade_seq = 0
        self.midday_done_for_session = False

        anchor = float(row[self._anchor_col()]) if pd.notna(row[self._anchor_col()]) else float(row["close"])
        grid_pct = float(row[self._grid_col()]) if pd.notna(row[self._grid_col()]) else np.nan
        if not np.isfinite(grid_pct):
            grid_pct = float(df[self._grid_col()].iloc[: i + 1].dropna().iloc[-1]) if df[self._grid_col()].iloc[: i + 1].dropna().any() else self.cfg.grid.pct
        layers = self._resolve_layers(df, i, anchor, grid_pct)
        self._reset_grid(ts, anchor, grid_pct, layers, trade_log, state, reason="NEW_SESSION", price_for_log=float(row["close"]), keep_layer=0)

    def on_bar(self, df: pd.DataFrame, i: int, state: dict, trade_log: List[dict], prev_row: Optional[pd.Series]) -> None:
        row = df.iloc[i]
        ts = row.name

        # New session handling + force close prior day.
        if self.current_session is None:
            self.initialize_session(df, i, row, state, trade_log)
        elif row["session_date"] != self.current_session:
            if self.cfg.risk.force_close_end_of_day and prev_row is not None and state["position"] > 0:
                px = float(prev_row[self.cfg.risk.price_field_for_execution])
                self._force_close(prev_row.name, px, state, trade_log, reason="SessionEnd")
            self.initialize_session(df, i, row, state, trade_log)

        # Optional intraday forced close at configured time.
        if self.cfg.risk.force_close_end_of_day and self.cfg.risk.session_close_time is not None:
            if ts.strftime("%H:%M") >= self.cfg.risk.session_close_time and state["position"] > 0:
                px = float(row[self.cfg.risk.price_field_for_execution])
                self._force_close(ts, px, state, trade_log, reason="ClockClose")
                return

        if self._maybe_hard_stop(row, i, state, trade_log):
            return
        self._maybe_reset(df, i, row, state, trade_log)
        self._maybe_trade(row, state, trade_log, i)


# ----------------------------- Backtest runner -----------------------------
def performance_stats(equity: pd.Series, bars_per_year: int = 252 * 78) -> Dict[str, float]:
    ret = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0 if len(equity) else 0.0
    if len(equity) > 1:
        years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0 if equity.iloc[0] > 0 else np.nan
    else:
        cagr = 0.0
    vol = ret.std() * math.sqrt(bars_per_year) if len(ret) > 1 else 0.0
    sharpe = (ret.mean() / ret.std() * math.sqrt(bars_per_year)) if ret.std() > 1e-12 else np.nan
    rolling_max = equity.cummax()
    dd = equity / rolling_max - 1.0
    maxdd = dd.min() if len(dd) else 0.0
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    return {
        "TotalReturn": float(total_return),
        "CAGR": float(cagr),
        "Vol": float(vol),
        "Sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "MaxDD": float(maxdd),
        "Calmar": float(calmar) if np.isfinite(calmar) else np.nan,
    }


def run_flexible_grid_backtest(raw_df: pd.DataFrame, cfg: Optional[BacktestConfig] = None) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float], List[dict], pd.DataFrame]:
    cfg = cfg or BacktestConfig()
    df = prepare_intraday_features(raw_df, cfg)

    state = {
        "cash": float(cfg.initial_cash),
        "position": 0.0,
        "avg_price": np.nan,
        "trades": 0,
        "fees_paid": 0.0,
        "target_size": cfg.units_per_layer,
    }
    trade_log: List[dict] = []
    strategy = FlexibleIntradayGrid(cfg)

    records = []
    prev_row = None
    for i in range(len(df)):
        row = df.iloc[i]
        strategy.on_bar(df, i, state, trade_log, prev_row)
        mark_px = float(row["close"])
        equity = state["cash"] + state["position"] * mark_px
        records.append({
            "cash": state["cash"],
            "position": state["position"],
            "avg_price": state["avg_price"],
            "equity": equity,
            "close": mark_px,
            "anchor": strategy.anchor,
            "grid_pct": strategy.grid_pct,
            "layer": strategy.layer,
            "max_layers": strategy.max_layers,
            "adx": row.get("adx", np.nan),
            "vwap": row.get("vwap", np.nan),
        })
        prev_row = row

    # Final close if needed
    if cfg.risk.force_close_end_of_day and state["position"] > 0 and len(df) > 0:
        last_ts = df.index[-1]
        last_px = float(df.iloc[-1][cfg.risk.price_field_for_execution])
        strategy._force_close(last_ts, last_px, state, trade_log, reason="FinalBar")
        records[-1]["cash"] = state["cash"]
        records[-1]["position"] = state["position"]
        records[-1]["avg_price"] = state["avg_price"]
        records[-1]["equity"] = state["cash"]

    res = pd.DataFrame(records, index=df.index)
    stats = performance_stats(res["equity"])
    final = {
        "final_equity": float(res["equity"].iloc[-1]) if len(res) else cfg.initial_cash,
        "cash": float(state["cash"]),
        "position": float(state["position"]),
        "avg_price": float(state["avg_price"]) if pd.notna(state["avg_price"]) else np.nan,
        "trades": int(state["trades"]),
        "fees_paid": float(state["fees_paid"]),
        "config": asdict(cfg),
    }
    return res, stats, final, trade_log, df


# ----------------------------- Example usage -----------------------------
if __name__ == "__main__":
    # Example assumes you already have a minute-bar DataFrame named df with:
    # datetime/open/high/low/close/volume and optional bid/ask/spread.

    print("grid_backtest_flexible.py loaded. Use run_flexible_grid_backtest(your_df, cfg).")
