# -*- coding: utf-8 -*-
"""
用 baostock 给 Elec/data/ 里的每只电力股 CSV 补充换手率(turn)列，
并更新 data_Elec/stocks.csv（含起止日期）。
路径通过 __file__ 动态推导，避免中文路径硬编码乱码问题。
"""
import os, time
import pandas as pd
import baostock as bs

# 从脚本自身位置推导所有路径（不依赖中文硬编码字符串）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      # .../Quant/Astock
ASTOCKS    = os.path.join(SCRIPT_DIR, "Astocks")
DATA_DIR   = os.path.join(ASTOCKS, "Elec", "dataTushare")
STOCKS_SRC = os.path.join(ASTOCKS, "Elec", "stocks.csv")

print("DATA_DIR  :", DATA_DIR)
print("STOCKS_SRC:", STOCKS_SRC)

def to_bs_code(tscode: str) -> str:
    code, mkt = tscode.split(".")
    return f"{mkt.lower()}.{code}"

# ── 1. 建立股票索引（ts_code → filepath, start, end）────────────────────────
records, file_map = [], {}
csv_files = [f for f in sorted(os.listdir(DATA_DIR)) if f.endswith(".csv")]
print(f"\n扫描到 {len(csv_files)} 个 CSV 文件")

for fname in csv_files:
    path = os.path.join(DATA_DIR, fname)
    df   = pd.read_csv(path, usecols=["ts_code", "trade_date"])
    tscode = df["ts_code"].iloc[0]
    records.append({
        "ts_code":    tscode,
        "code":       fname[:6],
        "start_date": str(int(df["trade_date"].min())),
        "end_date":   str(int(df["trade_date"].max())),
        "rows":       len(df),
    })
    file_map[tscode] = path

df_info = pd.DataFrame(records)
print(f"df_info 行数: {len(df_info)}, 列: {df_info.columns.tolist()}")

# 合并 name / weight
if os.path.exists(STOCKS_SRC):
    df_meta = pd.read_csv(STOCKS_SRC, dtype=str)
    print(f"STOCKS_SRC 列: {df_meta.columns.tolist()}")
    if "ts_code" in df_meta.columns:
        keep = [c for c in ["ts_code","name","weight","in_date"] if c in df_meta.columns]
        df_info = df_info.merge(df_meta[keep], on="ts_code", how="left")

df_info.to_csv(STOCKS_SRC, index=False, encoding="utf-8-sig")
print(f"\nstocks.csv 已保存（{len(df_info)} 只）")
print(df_info[["ts_code","start_date","end_date","rows"]].head(5).to_string(index=False))

# ── 2. baostock 批量下载换手率并合并 ─────────────────────────────────────────
lg = bs.login()
print(f"\nbaostock login: {lg.error_code} {lg.error_msg}")
print(f"开始下载换手率，共 {len(df_info)} 只...\n")

failed = []
total  = len(df_info)

for i, row in df_info.iterrows():
    tscode = row["ts_code"]
    path   = file_map.get(tscode)
    
    if path is None:
        failed.append(tscode)
        continue

    df_k = pd.read_csv(path)

    if "turnover_rate" in df_k.columns:
        print(f"[{i+1:03d}/{total}] {tscode}  skip(已有)")
        continue

    bs_code   = to_bs_code(tscode)
    s, e      = row["start_date"], row["end_date"]
    start_str = f"{s[:4]}-{s[4:6]}-{s[6:]}"
    end_str   = f"{e[:4]}-{e[4:6]}-{e[6:]}"

    rs    = bs.query_history_k_data_plus(
                bs_code, fields="date,turn",
                start_date=start_str, end_date=end_str,
                frequency="d", adjustflag="3")
    df_tr = rs.get_data()

    if df_tr is None or df_tr.empty:
        print(f"[{i+1:03d}/{total}] {tscode}  empty")
        failed.append(tscode)
        continue

    df_tr["trade_date"]    = df_tr["date"].str.replace("-", "").astype(int)
    df_tr["turnover_rate"] = pd.to_numeric(df_tr["turn"], errors="coerce")

    df_k["trade_date"] = df_k["trade_date"].astype(int)
    df_merged = df_k.merge(df_tr[["trade_date","turnover_rate"]], on="trade_date", how="left")
    df_merged.to_csv(path, index=False, encoding="utf-8-sig")

    ok = df_tr["turnover_rate"].notna().sum()
    name = row.get("name", "")
    print(f"[{i+1:03d}/{total}] {tscode} ({name})  OK  rows={len(df_merged)}  TR={ok}")

bs.logout()
print(f"\n── 完成 ── 失败: {failed if failed else '无'}")
