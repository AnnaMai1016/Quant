# -*- coding: utf-8 -*-
"""
电力行业股票 —— 方向1 & 2 分析
  散点图：pct_chg vs turnover_rate / 量比
  四象限：放量涨/跌、缩量涨/跌 → 次日收益分布
  滞后相关性：corr(indicator[t], pct_chg[t+k])  k=0..5
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid

def linregress_np(x, y):
    """返回 (slope, intercept, r, p_approx)，纯 numpy 实现。"""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return 0.0, 0.0, 0.0, 1.0
    mx, my = x.mean(), y.mean()
    ss_xy  = ((x - mx) * (y - my)).sum()
    ss_xx  = ((x - mx) ** 2).sum()
    ss_yy  = ((y - my) ** 2).sum()
    slope     = ss_xy / ss_xx if ss_xx else 0.0
    intercept = my - slope * mx
    r = ss_xy / np.sqrt(ss_xx * ss_yy) if ss_xx * ss_yy > 0 else 0.0
    t = r * np.sqrt(n - 2) / np.sqrt(max(1 - r**2, 1e-12))
    p = 2 / (1 + abs(t) * np.sqrt(2 / n))   # Gaussian tail 近似
    return float(slope), float(intercept), float(r), float(p)

def pearsonr_np(x, y):
    """Pearson r，纯 numpy。"""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if np.isfinite(r) else 0.0

warnings.filterwarnings("ignore")

# ── 路径（__file__ 推导，无中文硬编码）─────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "Astocks", "Elec", "dataTushare")
OUT_DIR    = os.path.join(SCRIPT_DIR, "Astocks", "Elec", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":        "Microsoft YaHei",
    "axes.unicode_minus": False,
    "figure.dpi":         150,
})

# ── 1. 加载全部数据 ───────────────────────────────────────────────────────────
print("加载数据...")
frames = []
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
    df = df.sort_values("trade_date").reset_index(drop=True)
    # 量比 = 当日量 / 5日滚动均量（shift 1 避免用当日数据）
    df["vol_ma5"]    = df["vol"].rolling(5, min_periods=3).mean().shift(1)
    df["vol_ratio"]  = df["vol"] / df["vol_ma5"]
    for col in ["pct_chg", "turnover_rate", "vol_ratio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)
print(f"全量数据: {len(all_df):,} 行，{all_df['ts_code'].nunique()} 只股票")

# ── 近一个月（最近 22 个交易日）─────────────────────────────────────────────
recent_dates = sorted(all_df["trade_date"].unique())[-22:]
month_df = all_df[all_df["trade_date"].isin(recent_dates)].copy()
month_df = month_df.dropna(subset=["pct_chg", "turnover_rate", "vol_ratio"])
print(f"近一个月数据: {len(month_df):,} 行  ({recent_dates[0].date()} ~ {recent_dates[-1].date()})")

# ── 2. 散点图 + 线性回归 ─────────────────────────────────────────────────────
print("\n绘制散点图...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("电力行业近一个月  量价散点图", fontsize=14, fontweight="bold")

for ax, xcol, xlabel in [
    (axes[0], "turnover_rate", "换手率 (%)"),
    (axes[1], "vol_ratio",     "量比 (当日量/5日均量)"),
]:
    x = month_df[xcol].clip(
        lower=month_df[xcol].quantile(0.01),
        upper=month_df[xcol].quantile(0.99)
    )
    y = month_df["pct_chg"].clip(-10, 10)

    # 按涨跌着色
    colors = np.where(y >= 0, "#d63031", "#00b894")
    ax.scatter(x, y, c=colors, alpha=0.35, s=14, linewidths=0)

    # 线性回归
    mask = x.notna() & y.notna()
    slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
    xrange = np.linspace(x[mask].min(), x[mask].max(), 200)
    ax.plot(xrange, slope * xrange + intercept, color="navy",
            lw=1.8, label=f"r={r:.3f}  p={p:.3f}")

    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("当日涨跌幅 (%)", fontsize=10)
    ax.set_title(f"{xlabel}  vs  当日涨跌幅", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "01_scatter_price_vs_indicators.png")
fig.savefig(p1, bbox_inches="tight", dpi=180)
plt.close()
print(f"  saved: {p1}")

# ── 3. 四象限分析（次日收益分布）────────────────────────────────────────────
print("\n计算四象限...")

# 构建 (date, ts_code) → next-day pct_chg
all_sorted = all_df.sort_values(["ts_code", "trade_date"])
all_sorted["next_pct"] = all_sorted.groupby("ts_code")["pct_chg"].shift(-1)

month_q = all_sorted[all_sorted["trade_date"].isin(recent_dates)].dropna(
    subset=["pct_chg", "turnover_rate", "vol_ratio", "next_pct"]
).copy()

# 量比中位数为分界
vr_med = month_q["vol_ratio"].median()
pc_zero = 0.0

def quadrant(row):
    up   = row["pct_chg"] >= pc_zero
    high = row["vol_ratio"] >= vr_med
    if up and high:   return "放量上涨"
    if up and not high: return "缩量上涨"
    if not up and high: return "放量下跌"
    return "缩量下跌"

month_q["quadrant"] = month_q.apply(quadrant, axis=1)
ORDER = ["放量上涨", "缩量上涨", "放量下跌", "缩量下跌"]
COLORS = ["#d63031", "#ff7675", "#00b894", "#55efc4"]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("电力行业近一个月  四象限分析（量比中位数分界）", fontsize=14, fontweight="bold")

# 左图：各象限次日平均收益 + 置信区间
ax = axes[0]
means, cis, counts = [], [], []
for q in ORDER:
    g = month_q[month_q["quadrant"] == q]["next_pct"]
    counts.append(len(g))
    means.append(g.mean())
    se = g.std() / np.sqrt(len(g)) if len(g) > 1 else 0
    cis.append(1.96 * se)

bars = ax.bar(ORDER, means, color=COLORS, alpha=0.85, width=0.6)
ax.errorbar(ORDER, means, yerr=cis, fmt="none", color="black", capsize=5, lw=1.5)
ax.axhline(0, color="gray", lw=0.8, ls="--")
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f"n={cnt}", ha="center", fontsize=8)
ax.set_ylabel("次日平均涨跌幅 (%)", fontsize=10)
ax.set_title("四象限  →  次日平均收益（95% CI）", fontsize=11)
ax.grid(True, axis="y", alpha=0.3)

# 右图：Box plot
ax2 = axes[1]
data_boxes = [month_q[month_q["quadrant"] == q]["next_pct"].clip(-10, 10).values for q in ORDER]
bp = ax2.boxplot(data_boxes, patch_artist=True, notch=False,
                 medianprops=dict(color="white", lw=2),
                 whiskerprops=dict(lw=1.2),
                 flierprops=dict(marker=".", markersize=3, alpha=0.4))
for patch, color in zip(bp["boxes"], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax2.set_xticklabels(ORDER, fontsize=9)
ax2.axhline(0, color="gray", lw=0.8, ls="--")
ax2.set_ylabel("次日涨跌幅 (%)", fontsize=10)
ax2.set_title("四象限  →  次日收益分布（箱形图）", fontsize=11)
ax2.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "02_quadrant_next_day_return.png")
fig.savefig(p2, bbox_inches="tight", dpi=180)
plt.close()
print(f"  saved: {p2}")

# 打印象限统计
print("\n【四象限统计】")
for q, m, c_i, n in zip(ORDER, means, cis, counts):
    print(f"  {q:8s}  n={n:4d}  次日均收益={m:+.3f}%  ±{c_i:.3f}%")

# ── 4. 滞后相关性 corr(indicator[t], pct_chg[t+k]) ─────────────────────────
print("\n计算滞后相关性（全历史数据）...")
LAGS = [0, 1, 2, 3, 5]
INDICATORS = {
    "换手率 (turnover_rate)": "turnover_rate",
    "量比 (vol_ratio)":      "vol_ratio",
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
fig.suptitle("电力行业  滞后相关性  corr(indicator[t], pct_chg[t+k])\n（逐股计算后取均值±1σ）",
             fontsize=13, fontweight="bold")

for ax, (label, col) in zip(axes, INDICATORS.items()):
    lag_corrs = {k: [] for k in LAGS}

    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["vol_ma5"]   = df["vol"].rolling(5, min_periods=3).mean().shift(1)
        df["vol_ratio"] = df["vol"] / df["vol_ma5"]
        for c in ["pct_chg", "turnover_rate", "vol_ratio"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        for k in LAGS:
            x = df[col]
            y = df["pct_chg"].shift(-k)   # future return k days later
            valid = x.notna() & y.notna()
            if valid.sum() < 30:
                continue
            r, _ = stats.pearsonr(x[valid], y[valid])
            lag_corrs[k].append(r)

    means_lag = [np.mean(lag_corrs[k]) if lag_corrs[k] else np.nan for k in LAGS]
    stds_lag  = [np.std(lag_corrs[k])  if lag_corrs[k] else np.nan for k in LAGS]

    ax.bar(LAGS, means_lag, color=["#0984e3" if m >= 0 else "#d63031" for m in means_lag],
           alpha=0.7, width=0.6, label="均值相关系数")
    ax.errorbar(LAGS, means_lag, yerr=stds_lag, fmt="none",
                color="black", capsize=6, lw=1.5, label="±1σ（跨股票）")
    ax.axhline(0, color="gray", lw=0.8, ls="--")

    for k, m, s in zip(LAGS, means_lag, stds_lag):
        if not np.isnan(m):
            ax.text(k, m + (0.003 if m >= 0 else -0.005),
                    f"{m:+.3f}", ha="center", fontsize=8.5, fontweight="bold")

    ax.set_xticks(LAGS)
    ax.set_xticklabels([f"t+{k}" for k in LAGS], fontsize=10)
    ax.set_xlabel("滞后期 k（交易日）", fontsize=10)
    ax.set_ylabel("Pearson 相关系数", fontsize=10)
    ax.set_title(f"{label}\n→ pct_chg[t+k]", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(-0.15, 0.15)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, "03_lag_correlation.png")
fig.savefig(p3, bbox_inches="tight", dpi=180)
plt.close()
print(f"  saved: {p3}")

# ── 5. 换手率分位数 → 后续累计收益（分层回测）─────────────────────────────
print("\n计算换手率分层 → 后续累计收益...")

# 对近一个月每日横截面：按换手率分三组（低/中/高），计算次日平均收益
results = []
for date in recent_dates[:-1]:   # 去掉最后一天（无次日数据）
    day_df = all_sorted[all_sorted["trade_date"] == date].dropna(
        subset=["turnover_rate", "next_pct"])
    if len(day_df) < 10:
        continue
    day_df = day_df.copy()
    day_df["tr_tier"] = pd.qcut(day_df["turnover_rate"], 3, labels=["低换手","中换手","高换手"])
    g = day_df.groupby("tr_tier", observed=True)["next_pct"].mean()
    g["date"] = date
    results.append(g)

tier_df = pd.DataFrame(results).set_index("date")

fig, ax = plt.subplots(figsize=(14, 5))
TIER_COLORS = {"低换手": "#00b894", "中换手": "#fdcb6e", "高换手": "#d63031"}
for tier, color in TIER_COLORS.items():
    if tier in tier_df.columns:
        cum = tier_df[tier].cumsum()
        ax.plot(tier_df.index, cum, marker="o", markersize=4,
                lw=1.5, color=color, label=f"{tier} (累计)")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_title("近一个月  换手率三分位 × 次日收益  累计走势", fontsize=12, fontweight="bold")
ax.set_ylabel("累计次日均收益 (%)", fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
plt.xticks(rotation=25, fontsize=8)
plt.tight_layout()
p4 = os.path.join(OUT_DIR, "04_turnover_tier_cumret.png")
fig.savefig(p4, bbox_inches="tight", dpi=180)
plt.close()
print(f"  saved: {p4}")

print("\n══════════════════════════════════")
print("全部分析完成，图片保存至:")
print(f"  {OUT_DIR}")
for p in [p1, p2, p3, p4]:
    print(f"  {os.path.basename(p)}")
