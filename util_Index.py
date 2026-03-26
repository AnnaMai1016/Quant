import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================Single Index =========================
WINDOW_MID = 30
# def historical_volatility(prices, window=20):
#     """年化历史波动率"""
#     returns = np.log(prices / prices.shift(1))   # log收益
#     hv = returns.rolling(window).std() * np.sqrt(252)  # .std过去widow收益的标准差, *sqrt(252)年化
#     return hv
def yang_zhang_volatility(open, high, low, close, window=WINDOW_MID):
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    
    overnight = np.log(open / close.shift(1))    # 隔夜跳空部分
    oc_vol = overnight.rolling(window).var()
    
    rs = (np.log(high / close) * np.log(high / open) +
          np.log(low / close) * np.log(low / open))    # 日内部分（Rogers-Satchell）
    rs_vol = rs.rolling(window).var()
    
    cc = np.log(close / open)    # 收盘价部分
    cc_vol = cc.rolling(window).var()
    
    hv = np.sqrt(oc_vol + k * cc_vol + (1 - k) * rs_vol) * np.sqrt(252)  # 年化
    return hv


def cal_atr(high, low, close, window=WINDOW_MID):
    """Avrage True Range, 网格间距一般用ATR%的1-2倍"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_abs = tr.rolling(window).mean()
    
    atr_pct = atr_abs / close  # (%) /current price
    return atr_abs, atr_pct


# ======== 波动范围 ==========
def bollinger_bandwidth(prices, window=WINDOW_MID, std_dev=2):
    """Bollinger Band Width""" 
    ma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    bandwidth = (upper - lower) / ma  # 归一化宽度
    return bandwidth

def price_range_ratio(high, low, close, window=WINDOW_MID*4):
    """N周最高最低价区间 / 当前价"""
    period_high = high.rolling(window).max()
    period_low = low.rolling(window).min()
    range_ratio = (period_high - period_low) / close
    return range_ratio
def price_percentile(close, window=252):   # 固定一年
    percentile = close.rolling(window).rank(pct=True)
    return percentile  # 0=历史最低, 1=历史最高
# ========================= =========================




# ========================= Indexes =========================
def volatility_indicators(open, high, low, close, figsize=(16, 20), draw=False):
    """
    Cal and Plot 6 volatility/range indicators in subplots with shared x-axis (time).
    """
    yz           = yang_zhang_volatility(open, high, low, close)
    atr_abs, atr_pct = cal_atr(high, low, close)
    bb           = bollinger_bandwidth(close)
    prr          = price_range_ratio(high, low, close)
    ppercent     = price_percentile(close)

    if draw:
        fig, axes = plt.subplots(6, 1, figsize=figsize, sharex=True)
        fig.suptitle("Volatility & Range Indicators", fontsize=15, fontweight="bold", y=1.01)

        plots = [(yz,       "Yang-Zhang\nVolatility",      "Yang-Zhang Volatility",     "steelblue"),
                (atr_abs,  "ATR\n(Absolute, price unit)",  "ATR abs $",                "darkorange"),
                (atr_pct,  "ATR\n(% of Close)",            "ATR (ATR ABS/Close)",                  "tomato"),
                (bb,       "Bollinger\nBandwidth",         "Bollinger Bandwidth",               "mediumseagreen"),
                (prr,      "Price Range\nRatio",           "Price Range Ratio",                    "mediumpurple"),
                (ppercent, "Price\nPercentile",            "Price Percentile (0~1)",       "sienna"),
            ]

        for ax, (series, ylabel, title, color) in zip(axes, plots):
            ax.plot(series.index, series.values, color=color, linewidth=1.2)
            # ax.set_ylabel(ylabel, fontsize=14, labelpad=8)
            ax.set_title(title, fontsize=14, loc="left", pad=3)
            ax.grid(axis="both", linestyle=":", alpha=0.5)
            ax.tick_params(axis="x", rotation=30)
            if ax != axes[-1]: plt.setp(ax.get_xticklabels(), visible=False)

        axes[-1].set_xlabel("Time", fontsize=14)
        plt.tight_layout()
        plt.show()

    return yz,  atr_abs, atr_pct,  bb,  prr,  ppercent