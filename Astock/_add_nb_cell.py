# -*- coding: utf-8 -*-
import json, os

NEW_SOURCE = """\
# turnover price T
import os, pandas as pd

DATA_DIR   = os.path.join(os.getcwd(), 'Astocks', 'Elec', 'dataTushare')
STOCKS_CSV = os.path.join(os.getcwd(), 'Astocks', 'Elec', 'stocks.csv')

# ── 加载全部股票数据 ───────────────────────────────────────────────────────
frames = []
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith('.csv'):
        continue
    df = pd.read_csv(os.path.join(DATA_DIR, fname),
                     usecols=['ts_code','trade_date','pct_chg','turnover_rate'])
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)
all_df['trade_date']    = pd.to_datetime(all_df['trade_date'].astype(str))
all_df['pct_chg']       = pd.to_numeric(all_df['pct_chg'],       errors='coerce')
all_df['turnover_rate'] = pd.to_numeric(all_df['turnover_rate'], errors='coerce')

# 最近5个交易日
last5  = sorted(all_df['trade_date'].unique())[-5:]
dates  = [str(d.date()) for d in last5]
print("最近5个交易日:", dates)
recent = all_df[all_df['trade_date'].isin(last5)]

# ── 条件1：过去5天换手率均值 > 3% ─────────────────────────────────────────
tr_mean      = recent.groupby('ts_code')['turnover_rate'].mean().rename('tr_mean_5d').reset_index()
df_turnover  = tr_mean[tr_mean['tr_mean_5d'] > 3].sort_values('tr_mean_5d', ascending=False)

# ── 条件2：过去5天任意一天涨跌幅 > 5% ────────────────────────────────────
max_pct  = recent.groupby('ts_code')['pct_chg'].max().rename('max_pct_5d').reset_index()
df_p     = max_pct[max_pct['max_pct_5d'] > 5].sort_values('max_pct_5d', ascending=False)

# ── 合并名称 ──────────────────────────────────────────────────────────────
df_meta     = pd.read_csv(STOCKS_CSV, dtype=str)[['ts_code','name','weight']]
df_turnover = df_turnover.merge(df_meta, on='ts_code', how='left')
df_p        = df_p.merge(df_meta, on='ts_code', how='left')

print(f"\\n换手率均值 >3%（过去5天）：{len(df_turnover)} 只")
print(f"任意一天涨跌幅 >5%（过去5天）：{len(df_p)} 只")

# ── 重合 ──────────────────────────────────────────────────────────────────
overlap_codes = set(df_turnover['ts_code']) & set(df_p['ts_code'])
df_overlap = (df_turnover[df_turnover['ts_code'].isin(overlap_codes)]
              .merge(df_p[['ts_code','max_pct_5d']], on='ts_code')
              .sort_values('tr_mean_5d', ascending=False))

print(f"\\n{'='*50}")
print(f"两者重合：{len(df_overlap)} 只")
print('='*50)
display(df_overlap[['ts_code','name','tr_mean_5d','max_pct_5d','weight']].reset_index(drop=True))

print("\\n── 换手率 >3% 全部 ──")
display(df_turnover[['ts_code','name','tr_mean_5d','weight']].reset_index(drop=True))

print("\\n── 涨跌幅 >5% 全部 ──")
display(df_p[['ts_code','name','max_pct_5d','weight']].reset_index(drop=True))
"""

nb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "A002230.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Remove old cell with same id if exists
nb["cells"] = [c for c in nb["cells"] if c.get("id") != "a1b2c3d4"]

new_cell = {
    "id": "a1b2c3d4",
    "cell_type": "code",
    "source": [NEW_SOURCE],
    "metadata": {},
    "outputs": [],
    "execution_count": None
}
nb["cells"].append(new_cell)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Cell added:", nb_path)
