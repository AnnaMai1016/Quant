"""
画出 8 个申万一级行业（2016-至今）的换手率和量比。
- 换手率：从申万研究所 API 获取
- 量比：从本地 CSV 计算（当日成交量 / 5日均量）
"""
import os, math, time, warnings
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
requests.packages.urllib3.disable_warnings()

# ── 配置 ─────────────────────────────────────────────────────────────────────
CSV_DIR    = r"C:\Users\junyu\OneDrive\桌面\Quant\Astock\Astocks"
KLINE_DIR  = os.path.join(CSV_DIR, "data_Primary_industry")
OUT_PATH   = r"C:\Users\junyu\OneDrive\桌面\Quant\Astock\Astocks\plots\sw1_turnover_volratio.png"
START_DATE = "2016-01-01"
END_DATE   = pd.Timestamp.today().strftime("%Y-%m-%d")
SMOOTH     = 20          # 换手率用 N 日滚动均线平滑显示

plt.rcParams.update({
    "font.family":        "Microsoft YaHei",
    "axes.unicode_minus": False,
    "figure.dpi":         150,
})

# 9 个目标行业
TARGETS = {
    "801750": "计算机",
    "801730": "电力设备",
    "801210": "社会服务",
    "801160": "公用事业",
    "801890": "机械设备",
    "801740": "国防军工",
    "801880": "汽车",
    "801030": "基础化工",
    "801050": "有色金属",
}

COLORS = [
    "#e17055", "#0984e3", "#00b894", "#6c5ce7",
    "#fdcb6e", "#d63031", "#74b9ff", "#55efc4",
]

# ── 1. 下载换手率（申万研究所 API）────────────────────────────────────────────
SW_URL = "https://www.swsresearch.com/institute-sw/api/index_analysis/index_analysis_report/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}

def fetch_turnover(code: str, name: str) -> pd.DataFrame:
    cache_path = os.path.join(CSV_DIR, f"{code}_turnover.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["date"])
        print(f"  [cache] {name} 换手率 ({len(df)} 行)")
        return df

    params = {
        "page_size": "500",
        "index_type": "一级行业",
        "start_date": START_DATE,
        "end_date": END_DATE,
        "type": "DAY",
        "swindexcode": code,
    }
    all_rows = []
    r = requests.get(SW_URL, params={**params, "page": "1"}, headers=HEADERS,
                     verify=False, timeout=30)
    data = r.json()["data"]
    total_pages = math.ceil(data["count"] / 500)
    all_rows.extend(data["results"])
    for pg in range(2, total_pages + 1):
        r = requests.get(SW_URL, params={**params, "page": str(pg)},
                         headers=HEADERS, verify=False, timeout=30)
        all_rows.extend(r.json()["data"]["results"])
        time.sleep(0.25)

    df = pd.DataFrame(all_rows)[["swindexcode", "bargaindate", "turnoverrate"]]
    df.columns = ["code", "date", "turnover"]
    df["date"]     = pd.to_datetime(df["date"].str[:10])
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
    df = df[df["date"] >= START_DATE].sort_values("date").reset_index(drop=True)
    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    print(f"  [save] {name} 换手率 ({len(df)} 行)")
    return df


# ── 2. 读取量数据，计算量比（= 当日量 / 5日均量）────────────────────────────
def load_vol_ratio(code: str, name: str, window: int = 5) -> pd.DataFrame:
    # 文件名格式：801750.SI_计算机.csv，存放在 data_Primary_industry 子目录
    for f in os.listdir(KLINE_DIR):
        if f.startswith(code) and f.endswith(".csv"):
            path = os.path.join(KLINE_DIR, f)
            df = pd.read_csv(path)
            df.columns = ["ts_code", "trade_date", "open", "close", "high", "low", "vol", "amount"]
            df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
            df = df[df["trade_date"] >= START_DATE].sort_values("trade_date").reset_index(drop=True)
            df["vol"] = pd.to_numeric(df["vol"], errors="coerce")
            df["vol_ma"] = df["vol"].rolling(window, min_periods=1).mean()
            df["vol_ratio"] = df["vol"] / df["vol_ma"].shift(1)
            df = df[["trade_date", "vol", "vol_ratio"]].rename(columns={"trade_date": "date"})
            print(f"  [vol]  {name} 量比 ({len(df)} 行)")
            return df
    print(f"  [warn] {name}: 未找到本地 CSV")
    return pd.DataFrame(columns=["date", "vol", "vol_ratio"])


# ── 下载/读取所有行业数据 ────────────────────────────────────────────────────
print("── 获取换手率数据 ──")
turnover_dict = {}
for code, name in TARGETS.items():
    turnover_dict[name] = fetch_turnover(code, name)
    time.sleep(0.2)

print("\n── 计算量比 ──")
vol_dict = {}
for code, name in TARGETS.items():
    vol_dict[name] = load_vol_ratio(code, name)


# ── 3. 画图 ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(22, 14), sharex=False)
fig.suptitle(
    f"申万一级行业  换手率 & 量比  {START_DATE} ~ {END_DATE}",
    fontsize=15, fontweight="bold", y=0.98
)

ax_tr, ax_vr = axes

for (name, color) in zip(TARGETS.values(), COLORS):
    # ── 换手率 ──
    df_t = turnover_dict.get(name)
    if df_t is not None and not df_t.empty:
        s = df_t.set_index("date")["turnover"]
        s_smooth = s.rolling(SMOOTH, min_periods=1).mean()
        ax_tr.plot(s_smooth.index, s_smooth.values,
                   color=color, lw=1.2, label=name, alpha=0.9)
        ax_tr.fill_between(s.index, s.values, alpha=0.04, color=color)

    # ── 量比 ──
    df_v = vol_dict.get(name)
    if df_v is not None and not df_v.empty:
        s2 = df_v.set_index("date")["vol_ratio"].replace([np.inf, -np.inf], np.nan)
        s2_smooth = s2.rolling(SMOOTH, min_periods=1).mean()
        ax_vr.plot(s2_smooth.index, s2_smooth.values,
                   color=color, lw=1.2, label=name, alpha=0.9)
        ax_vr.fill_between(s2.index, s2.values.clip(0, 5),
                            alpha=0.04, color=color)

# ── 格式 ──
for ax, title, ylabel in [
    (ax_tr, f"换手率 % （{SMOOTH}日均线）", "换手率 (%)"),
    (ax_vr, f"量比 = 当日量 / 5日均量（{SMOOTH}日均线）", "量比"),
]:
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, loc="upper left", ncol=4, framealpha=0.8)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.tick_params(axis="x", labelsize=8)

ax_tr.axhline(0, color="gray", lw=0.5, ls="--")
ax_vr.axhline(1, color="gray", lw=1.0, ls="--")   # 量比=1 基准线
ax_vr.set_ylim(bottom=0, top=min(ax_vr.get_ylim()[1], 4))

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT_PATH, bbox_inches="tight", dpi=180)
plt.close()
print(f"\n── 完成，图片保存至：\n   {OUT_PATH}")