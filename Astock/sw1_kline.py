"""
下载申万一级行业指数日K线 (2016-01-01 至今)，保存 CSV，并绘制 K线+布林带+MA / MACD 图。
数据源：akshare index_hist_sw（免费，无需积分）
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import akshare as ak

warnings.filterwarnings("ignore")

# ── 配置 ─────────────────────────────────────────────────────────────────────
INDEX_BASIC = r"C:\Users\junyu\OneDrive\桌面\Quant\Astock\index_basic_ts_20260519.csv"
OUT_DIR     = r"C:\Users\junyu\OneDrive\桌面\Quant\Astock\Astocks\data"
PLOT_DIR    = os.path.join(OUT_DIR, "plots")
START_DATE  = "2016-01-01"
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":        "Microsoft YaHei",
    "axes.unicode_minus": False,
    "figure.dpi":         150,
})

# ── 申万一级行业指数（过滤退市）────────────────────────────────────────────
df_basic = pd.read_csv(INDEX_BASIC)
sw1 = df_basic[
    (df_basic["category"] == "一级行业指数") &
    (~df_basic["name"].str.contains("退市"))
][["ts_code", "name"]].reset_index(drop=True)

# 提取纯数字代码（akshare 用 "801010" 不含 ".SI"）
sw1["code"] = sw1["ts_code"].str.replace(r"\.\w+$", "", regex=True)
# 清理文件名
sw1["clean_name"] = sw1["name"].str.replace(r"\(申万\)", "", regex=True).str.replace("/", "-")

print(f"共 {len(sw1)} 个申万一级行业指数")
print(sw1[["ts_code", "name"]].to_string(index=False))
print()


# ── 指标计算 ─────────────────────────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    df["ma5"]      = c.rolling(5).mean()
    df["ma10"]     = c.rolling(10).mean()
    df["ma20"]     = c.rolling(20).mean()
    df["bb_mid"]   = c.rolling(20).mean()
    df["bb_std"]   = c.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    ema12          = c.ewm(span=12, adjust=False).mean()
    ema26          = c.ewm(span=26, adjust=False).mean()
    df["macd_dif"] = ema12 - ema26
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd_bar"] = (df["macd_dif"] - df["macd_dea"]) * 2
    return df


# ── K 线图 ────────────────────────────────────────────────────────────────────
def plot_candles(ax, df: pd.DataFrame):
    x = np.arange(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        color = "#d63031" if row["close"] >= row["open"] else "#00b894"
        ax.plot([i, i], [row["low"], row["high"]], color=color, lw=0.6)
        body_bot = min(row["open"], row["close"])
        body_h   = max(abs(row["close"] - row["open"]), row["close"] * 0.001)
        ax.bar(i, body_h, bottom=body_bot, color=color, width=0.7, linewidth=0)

    ax.plot(x, df["ma5"],      color="#0984e3", lw=0.9,  label="MA5")
    ax.plot(x, df["ma10"],     color="#fdcb6e", lw=0.9,  label="MA10")
    ax.plot(x, df["ma20"],     color="#a29bfe", lw=0.9,  label="MA20")
    ax.plot(x, df["bb_upper"], color="#55efc4", lw=0.8,  ls="--", label="BB上轨")
    ax.plot(x, df["bb_mid"],   color="#55efc4", lw=0.7,  ls=":",  label="BB中轨")
    ax.plot(x, df["bb_lower"], color="#55efc4", lw=0.8,  ls="--", label="BB下轨")
    ax.fill_between(x, df["bb_upper"], df["bb_lower"], color="#55efc4", alpha=0.07)

    step   = max(1, len(df) // 15)
    ticks  = x[::step]
    labels = [df.index[i].strftime("%Y-%m") for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, fontsize=6)
    ax.legend(fontsize=6, loc="upper left", ncol=3)
    ax.set_xlim(-1, len(df))


# ── MACD 图 ───────────────────────────────────────────────────────────────────
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


# ── 主流程 ────────────────────────────────────────────────────────────────────
failed = []

for _, row in sw1.iterrows():
    ts_code    = row["ts_code"]
    code       = row["code"]
    clean_name = row["clean_name"]
    fname      = f"{ts_code}_{clean_name}"
    csv_path   = os.path.join(OUT_DIR, f"{fname}.csv")
    plot_path  = os.path.join(PLOT_DIR, f"{fname}.png")

    # ── 下载 / 读缓存 ────────────────────────────────────────────────────
    if os.path.exists(csv_path):
        raw = pd.read_csv(csv_path)
        print(f"[cache]  {fname}  ({len(raw)} 行)")
    else:
        try:
            raw = ak.index_hist_sw(symbol=code, period="day")
            time.sleep(0.5)
        except Exception as e:
            print(f"[error]  {fname}: {e}")
            failed.append(fname)
            continue

        if raw is None or raw.empty:
            print(f"[empty]  {fname}")
            failed.append(fname)
            continue

        # 统一列名（akshare 返回位置固定：代码,日期,开,收,高,低,量,额）
        raw.columns = ["ts_code", "trade_date", "open", "close", "high", "low", "vol", "amount"]
        raw.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[save]   {fname}  ({len(raw)} 行)")

    # ── 准备 DataFrame ───────────────────────────────────────────────────
    df = raw.copy()
    df.columns = ["ts_code", "trade_date", "open", "close", "high", "low", "vol", "amount"]
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
    df = df[df["trade_date"] >= START_DATE].sort_values("trade_date").reset_index(drop=True)
    df.set_index("trade_date", inplace=True)
    for col in ["open", "close", "high", "low"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open", "close", "high", "low"], inplace=True)

    if len(df) < 30:
        print(f"[skip]   {fname}: 数据不足 ({len(df)} 行)")
        continue

    df = calc_indicators(df)

    # ── 绘图 ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 10))
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.4)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    start_str = df.index[0].strftime("%Y-%m-%d")
    end_str   = df.index[-1].strftime("%Y-%m-%d")
    fig.suptitle(f"{clean_name}  ({ts_code})   日K线   {start_str} ~ {end_str}",
                 fontsize=13, fontweight="bold", y=0.98)

    plot_candles(ax1, df)
    ax1.set_title("K线 + Bollinger Bands(20,2) + MA5 / MA10 / MA20", fontsize=9, pad=4)
    ax1.set_ylabel("指数点位", fontsize=8)
    ax1.grid(True, alpha=0.25, linestyle="--")

    plot_macd(ax2, df)
    ax2.set_title("MACD (12, 26, 9)", fontsize=9, pad=4)
    ax2.set_ylabel("MACD", fontsize=8)
    ax2.grid(True, alpha=0.25, linestyle="--")

    fig.savefig(plot_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"[plot]   {plot_path}")

print("\n── 全部完成 ──")
if failed:
    print("失败：", failed)