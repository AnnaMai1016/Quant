"""
申万二级行业"电力"(801161) 全部 A 股成分股
- 成分股列表：akshare index_component_sw（单次调用）
- 日K线下载：tushare pro.daily（更稳定，自带重试）
- 保存 CSV 到 data_Elec/
- 画图：上图 K线+BB+MA5/10/20+资金净流入代理（右轴） / 下图 MACD
"""
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import akshare as ak
import tushare as ts

warnings.filterwarnings("ignore")

TOKEN = os.environ["TUSHARE_TOKEN"]   # 本地环境变量: TUSHARE_TOKEN

# ── 配置 ─────────────────────────────────────────────────────────────────────
OUT_DIR    = r"C:\Users\junyu\OneDrive\桌面\Quant\Astock\Astocks\Elec"
PLOT_DIR   = os.path.join(OUT_DIR, "plots")
DATA_DIR   = os.path.join(OUT_DIR, "data")
START_DATE = "20160101"
END_DATE   = pd.Timestamp.today().strftime("%Y%m%d")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":        "Microsoft YaHei",
    "axes.unicode_minus": False,
    "figure.dpi":         150,
})

pro = ts.pro_api(TOKEN)


# ── 1. 获取电力行业成分股（申万二级 801161）────────────────────────────────
print("获取电力行业成分股 (801161)...")
df_comp = ak.index_component_sw(symbol="801161")
df_comp.columns = ["rank", "code", "name", "weight", "in_date"]
df_comp["code"] = df_comp["code"].astype(str).str.zfill(6)
# 判断市场后缀：60/68 开头→.SH，否则→.SZ
def ts_code(c):
    return c + (".SH" if c[:2] in ("60", "68", "90") else ".SZ")
df_comp["ts_code"] = df_comp["code"].apply(ts_code)
df_comp.to_csv(os.path.join(OUT_DIR, "stocks.csv"), index=False, encoding="utf-8-sig")

print(f"共 {len(df_comp)} 只成分股")
print(df_comp[["ts_code","name","weight"]].to_string(index=False))
print()


# ── 2. tushare 下载（带重试）────────────────────────────────────────────────
def fetch_daily(tscode: str, max_retry: int = 3) -> pd.DataFrame:
    for attempt in range(max_retry):
        try:
            df = pro.daily(ts_code=tscode, start_date=START_DATE, end_date=END_DATE)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            wait = 2 ** attempt
            print(f"  retry {attempt+1}/{max_retry} ({e}) wait {wait}s", end=" ")
            time.sleep(wait)
    return pd.DataFrame()


# ── 3. 指标计算 ───────────────────────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    df["ma5"]      = c.rolling(5).mean()
    df["ma10"]     = c.rolling(10).mean()
    df["ma20"]     = c.rolling(20).mean()
    df["bb_mid"]   = c.rolling(20).mean()
    df["bb_upper"] = df["bb_mid"] + 2 * c.rolling(20).std()
    df["bb_lower"] = df["bb_mid"] - 2 * c.rolling(20).std()

    ema12          = c.ewm(span=12, adjust=False).mean()
    ema26          = c.ewm(span=26, adjust=False).mean()
    df["macd_dif"] = ema12 - ema26
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd_bar"] = (df["macd_dif"] - df["macd_dea"]) * 2

    # 资金净流入代理 = 成交额(万元→亿) × (收-开)/(高-低+ε)
    # tushare amount 单位：千元，转亿元除以1e5
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    df["net_inflow"]      = df["amount"] / 1e5 * (df["close"] - df["open"]) / hl
    df["net_inflow_ma10"] = df["net_inflow"].rolling(10, min_periods=1).mean()
    return df


# ── 4. 绘图函数 ───────────────────────────────────────────────────────────────
def plot_candles(ax, ax_r, df: pd.DataFrame):
    x = np.arange(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        color = "#d63031" if row["close"] >= row["open"] else "#00b894"
        ax.plot([i, i], [row["low"], row["high"]], color=color, lw=0.6)
        body_bot = min(row["open"], row["close"])
        body_h   = max(abs(row["close"] - row["open"]), row["close"] * 0.001)
        ax.bar(i, body_h, bottom=body_bot, color=color, width=0.7, linewidth=0)

    ax.plot(x, df["ma5"],      color="#0984e3", lw=0.9, label="MA5")
    ax.plot(x, df["ma10"],     color="#fdcb6e", lw=0.9, label="MA10")
    ax.plot(x, df["ma20"],     color="#a29bfe", lw=0.9, label="MA20")
    ax.plot(x, df["bb_upper"], color="#55efc4", lw=0.8, ls="--", label="BB上轨")
    ax.plot(x, df["bb_mid"],   color="#55efc4", lw=0.7, ls=":",  label="BB中轨")
    ax.plot(x, df["bb_lower"], color="#55efc4", lw=0.8, ls="--", label="BB下轨")
    ax.fill_between(x, df["bb_upper"], df["bb_lower"], color="#55efc4", alpha=0.07)

    ni = df["net_inflow"].values
    ni_colors = np.where(ni >= 0, "#ff7675", "#00cec9")
    ax_r.bar(x, ni, color=ni_colors, width=0.7, linewidth=0, alpha=0.45, label="净流入代理")
    ax_r.plot(x, df["net_inflow_ma10"], color="#e17055", lw=1.0, label="净流入MA10")
    ax_r.axhline(0, color="gray", lw=0.5, ls="--")
    ax_r.set_ylabel("净流入代理(亿)", fontsize=7, color="#e17055")
    ax_r.tick_params(axis="y", labelsize=6, labelcolor="#e17055")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper left", ncol=4)

    step   = max(1, len(df) // 15)
    ticks  = x[::step]
    labels = [df.index[i].strftime("%Y-%m") for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, fontsize=6)
    ax.set_xlim(-1, len(df))
    ax_r.set_xlim(-1, len(df))


def plot_macd(ax, df: pd.DataFrame):
    x      = np.arange(len(df))
    colors = np.where(df["macd_bar"] >= 0, "#d63031", "#00b894")
    ax.bar(x, df["macd_bar"], color=colors, width=0.7, linewidth=0, label="MACD柱")
    ax.plot(x, df["macd_dif"], color="#0984e3", lw=0.9, label="DIF")
    ax.plot(x, df["macd_dea"], color="#fdcb6e", lw=0.9, label="DEA")
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    step   = max(1, len(df) // 15)
    ticks  = x[::step]
    labels = [df.index[i].strftime("%Y-%m") for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, fontsize=6)
    ax.legend(fontsize=6, loc="upper left", ncol=3)
    ax.set_xlim(-1, len(df))


# ── 5. 主流程 ─────────────────────────────────────────────────────────────────
failed = []
total  = len(df_comp)

for idx, row in df_comp.iterrows():
    code    = row["code"]
    tscode  = row["ts_code"]
    name    = row["name"].replace("/", "-").replace("*", "").replace("ST", "ST")
    fname   = f"{code}_{name}"
    csv_path  = os.path.join(DATA_DIR,  f"{fname}.csv")
    plot_path = os.path.join(PLOT_DIR, f"{fname}.png")

    print(f"[{idx+1:03d}/{total}] {fname}", end="  ")

    # ── 下载 / 缓存 ──────────────────────────────────────────────────────
    if os.path.exists(csv_path):
        raw = pd.read_csv(csv_path)
        print(f"cache({len(raw)}行)", end="  ")
    else:
        raw = fetch_daily(tscode)
        time.sleep(0.12)          # ~8 calls/s，tushare 限流下的安全速率

        if raw.empty:
            print("empty/failed")
            failed.append(fname)
            continue

        # tushare daily 列：ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount
        raw.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"saved({len(raw)}行)", end="  ")

    # ── 准备数据 ─────────────────────────────────────────────────────────
    df = raw.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
    df = df.sort_values("trade_date").reset_index(drop=True)
    df.set_index("trade_date", inplace=True)
    for col in ["open","close","high","low","amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open","close","high","low"], inplace=True)

    if len(df) < 30:
        print("skip(数据不足)")
        continue

    df = calc_indicators(df)

    # ── 绘图 ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 10))
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.4)
    ax1 = fig.add_subplot(gs[0])
    ax2 = ax1.twinx()
    ax3 = fig.add_subplot(gs[1])

    start_str = df.index[0].strftime("%Y-%m-%d")
    end_str   = df.index[-1].strftime("%Y-%m-%d")
    fig.suptitle(
        f"{name}（{tscode}）  申万电力  日K线  {start_str} ~ {end_str}",
        fontsize=13, fontweight="bold", y=0.98
    )

    plot_candles(ax1, ax2, df)
    ax1.set_title(
        "K线 + Bollinger Bands(20,2) + MA5/10/20  ｜  右轴：资金净流入代理 & MA10",
        fontsize=9, pad=4
    )
    ax1.set_ylabel("股价 (元)", fontsize=8)
    ax1.grid(True, alpha=0.2, linestyle="--", zorder=0)

    plot_macd(ax3, df)
    ax3.set_title("MACD (12, 26, 9)", fontsize=9, pad=4)
    ax3.set_ylabel("MACD", fontsize=8)
    ax3.grid(True, alpha=0.2, linestyle="--")

    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("plot OK")

print(f"\n── 全部完成 ── 共{total}只，失败{len(failed)}只")
if failed:
    print("失败：", failed)
