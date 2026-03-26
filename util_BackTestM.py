''''
    baostock
'''
import baostock as bs
import pandas as pd
import numpy as np


# ========================= Data Prepare =========================
'''
http://www.baostock.com/mainContent?file=stockKData.md
'''
def extract_data(stockIndex="sz.000713", start_date='2024-01-01', end_date='2024-12-31', 
                 frequency="d",  adjustflag="3"):
    '''sh,sz
       frequency: d (default daily), w,m, 5=5min, 15=15min, 30=30min, 60=60min
       adjustflag: default3=不复权(实盘), 1=后复权(), 2=前复权'''
    lg = bs.login() 
    if int(lg.error_code)!=0 or str(lg.error_msg)!='success':
        print(f'login respond error_code:{lg.error_code}, error_msg:{lg.error_msg}')

    # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
    # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
    # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
    rs = bs.query_history_k_data_plus(stockIndex, # "sh.600000" sh>500, sz<500
        "date,code,open,high,low,close,volume,isST", #preclose,amount,adjustflag,turn,tradestatus,pctChg,isST",
        start_date=start_date, end_date=end_date,
        frequency=frequency, adjustflag=adjustflag)
    if int(rs.error_code)!=0 or str(rs.error_msg)!='success':
        print(f'{stockIndex}, query_history_k_data_plus respond error_code={rs.error_code}, error_msg={rs.error_msg}')

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    df = pd.DataFrame(data_list, columns=rs.fields)

    # df.to_csv("D:\\history_A_stock_k_data.csv", index=False)
    bs.logout()

    df.columns = df.columns.str.lower()
    cols = ['open', 'low', 'high', 'close', 'volume']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    # df[cols] = df[cols].astype(float)    # data cleaning

    return df
# ========================= =========================




# ========================= Trend Following =========================
def buy(state: dict, price: float, size: float = 1.0, fee: float = 0.0) -> None:
    """
    Execute a buy (long entry). Simplified: all-in size units (could be shares).
    state holds: cash, position, entry_price, trades, fees_paid
    """
    if state["position"] != 0:
        return  # already in position (keep it simple)

    cost = price * size
    if cost > state["cash"]:
        # not enough cash; scale down
        size = state["cash"] / price
        cost = price * size

    trade_fee = cost * fee
    state["cash"] -= (cost + trade_fee)
    state["position"] = size
    state["entry_price"] = price
    state["trades"] += 1
    state["fees_paid"] += trade_fee


def sell(state: dict, price: float, fee: float = 0.0) -> None:
    """
    Execute a sell (exit long). Simplified: close entire position.
    """
    if state["position"] == 0:
        return

    proceeds = price * state["position"]
    trade_fee = proceeds * fee
    state["cash"] += (proceeds - trade_fee)
    state["position"] = 0.0
    state["entry_price"] = np.nan
    state["trades"] += 1
    state["fees_paid"] += trade_fee


class Strategy_TrendFollowing:
    """
    Donchian breakout + ATR trailing stop (long-only minimal).
    - Entry: close > upper(N) computed on previous bars
    - Exit : close < lower(M) or close < peak - k*ATR
    """

    def __init__(self, entry_n=55, exit_n=20, atr_n=14, atr_k=2.0, warmup=None):
        '''
            entry_n: Donchian entry window, 20-daily, 55-classical haigui
            exit_n: Donchian exit window
            atr_n: market volatility, Vmax-Vmin (average true range)
            atr_k: ATR multiple used to set the trailing stop distance
        '''
        self.entry_n = int(entry_n)
        self.exit_n = int(exit_n)
        self.atr_n = int(atr_n)
        self.atr_k = float(atr_k)
        self.warmup = warmup if warmup is not None else max(self.entry_n, self.exit_n, self.atr_n) + 2

        # internal state during backtest
        self.peak = -np.inf

    @staticmethod
    def _calc_atr(df: pd.DataFrame, n: int) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / n, adjust=False).mean()  # Wilder RMA approx
        return atr

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["upper"] = out["high"].rolling(self.entry_n).max().shift(1)
        out["lower"] = out["low"].rolling(self.exit_n).min().shift(1)
        out["atr"] = self._calc_atr(out, self.atr_n)
        return out

    def on_bar(self, i: int, row: pd.Series, state: dict, fee: float) -> None:
        """
        Called each bar. Decide buy/sell based on row + state.
        """
        c = float(row["close"])
        upper = row["upper"]
        lower = row["lower"]
        atr = row["atr"]

        # Warmup / missing indicators
        if i < self.warmup or np.isnan(upper) or np.isnan(lower) or np.isnan(atr):
            return

        in_pos = state["position"] != 0

        if not in_pos:
            # Entry
            if c > float(upper):
                buy(state, price=c, size=state["target_size"], fee=fee)
                self.peak = c
        else:
            # Update peak for trailing stop
            if c > self.peak:
                self.peak = c

            exit_by_channel = c < float(lower)
            exit_by_atr = c < (self.peak - self.atr_k * float(atr))

            if exit_by_channel or exit_by_atr:
                sell(state, price=c, fee=fee)
                self.peak = -np.inf
