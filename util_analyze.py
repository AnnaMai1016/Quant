import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



# ========================== save trade log ==========================
def trades_to_df(trade_log, csv_file, savelog=False):
    df = pd.DataFrame(trade_log)
    df["notional"] = df["price"] * df["size"]
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["time", "trade_time"]).reset_index(drop=True)
    if savelog:
        df.to_csv(csv_file, index=False);   print(csv_file, ' saved')
    return df




# =============================== Draw func ===============================
def plot_ohlc_box_with_volume(df, stockIndex,
                              start="2004-08-19", end="2013-03-01", time_period=30, trade_log=None,
                              figuresize=(12, 6)):
    df.index = pd.to_datetime(df["date"])
    data = df.loc[start:end]

    if len(data) < time_period:
        raise ValueError("plot_ohlc_box_with_volume: time_period is too long for the selected date range.")

    # -------- 聚合 OHLC --------
    ohlc_list, vol_list, dates = [],[],[]

    for i in range(0, len(data), time_period):
        chunk = data.iloc[i:i+time_period]
        if len(chunk) == time_period:

            o = chunk["open"].iloc[0]
            c = chunk["close"].iloc[-1]
            h = chunk["high"].max()
            l = chunk["low"].min()
            v = chunk["volume"].sum()

            ohlc_list.append((o, c, h, l))
            vol_list.append(v)
            dates.append(chunk.index[0])

    fig, ax1 = plt.subplots(figsize=figuresize)
    plt.rcParams.update({"font.size": 9,  "axes.titlesize": 14,  "axes.labelsize": 12,  "xtick.labelsize": 10,  "ytick.labelsize": 10,  "legend.fontsize": 10})
    x = np.arange(len(ohlc_list))
    # date_to_x = {d: i for i, d in enumerate(dates)}
    # -------- 画 OHLC box --------
    for i, (o, c, h, l) in enumerate(ohlc_list):
        ax1.plot([i, i], [l, h], color="gray", linewidth=1)

        # open / close
        lower = min(o, c)
        height = abs(c - o)
        rect = plt.Rectangle((i - 0.3, lower),
                             0.6,
                             height if height != 0 else 0.01,
                             color="green" if c >= o else "red",alpha=0.6)
        ax1.add_patch(rect)

    ax1.set_ylabel("Price")
    ax1.set_xticks(x[::max(1, len(x)//12)])
    ax1.set_xticklabels(
        [dates[i].strftime("%Y-%m-%d") for i in x[::max(1, len(x)//12)]],
        rotation=45)
    ax1.grid(True)

    # -------- volume --------
    ax2 = ax1.twinx()
    ax2.bar(x, vol_list, alpha=0.3)
    ax2.set_ylabel("volume")

    plt.title(f"{stockIndex} OHLC per {time_period}days\n{start} to {end}")

    # -------- trade log --------
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    trade_log_filtered = [t for t in trade_log if start <= pd.Timestamp(t["time"]) <= end]
    for trade in trade_log_filtered:
        t, p, size, type = trade["time"], trade["price"], trade["size"], trade["type"]

        box_index = None
        for i, d in enumerate(dates):
            if d <= t:
                box_index = i
            else:
                break

        if box_index is None:
            continue

        if type == "BUY":    # BUY
            pn=trade["next_buy"]
            ax1.scatter(box_index, p, marker='+', s=size/4, color='green')  # "^" r"$\uparrow$"
            ax1.text(box_index, p, f"+{p:.2f}({pn:.2f})", color='k')
        elif type == "SELL":   # -1: Sell
            pn = trade["next_sell"]
            ax1.scatter(box_index, p, marker='+', s=size/4, color='red')  #"v" r"$\downarrow$"
            ax1.text(box_index, p, f"-{p:.2f}({pn:.2f})", color='k')
        elif type in ["SELL[Hard Stop1]", "SELL[Hard Stop2]"]:
            ax1.scatter(box_index, p, marker="v", s=size/20, facecolors='none', edgecolors='red')
            ax1.text(box_index, p, f"-{p:.2f}({type[-11:-1]})", color='k')
            common = dict(linestyle="-", color='blue')
            ax1.plot([box_index-5, box_index+5], [trade["next_sell"]]*2, **common)
            ax1.plot([box_index-5, box_index+5], [trade["next_buy"]]*2,  **common)
            ax1.plot([box_index]*2,              [trade["next_buy"], trade["next_sell"]], **common)
        elif type == "[RESET ANCHOR]":
            pn = trade["anchor"]
            ax1.plot([box_index-10, box_index+10], [pn,pn], linestyle=":", color='gray')
            print(box_index, pn)
            ax1.scatter([box_index], [pn], marker='*',color='gray')
            ax1.text(box_index, pn, f"RESET: {pn:.2f}", color='gray')
        elif type == "[BUY]LAYER==15":
            pn = trade["anchor"]
            ax1.scatter([box_index], [pn], marker='*',color='green')  #new anchor
            ax1.text(box_index, pn, f"{pn:.2f}[BUY]lmax", color='gray')  # 
            common = dict(linestyle="-", color='blue')
            ax1.plot([box_index-5, box_index+5], [trade["next_sell"]]*2, **common)
            ax1.plot([box_index-5, box_index+5], [trade["next_buy"]]*2,  **common)
            ax1.plot([box_index]*2,              [trade["next_buy"], trade["next_sell"]], **common)
        elif type == "[SELL]LAYER==0":
            pn = trade["anchor"]
            ax1.scatter([box_index], [pn], marker='*',color='red')
            # ax1.plot([box_index-10, box_index+10], [c,c], linestyle=":", color='red')
            ax1.text(box_index, pn, f"{pn:.2f}[SELL]l0", color='gray')
            common = dict(linestyle="-", color='blue')
            ax1.plot([box_index-5, box_index+5], [trade["next_sell"]]*2, **common)
            ax1.plot([box_index-5, box_index+5], [trade["next_buy"]]*2,  **common)
            ax1.plot([box_index]*2,              [trade["next_buy"], trade["next_sell"]], **common)

    plt.tight_layout()
    plt.show()



def plot_equity_cash_position(res, stats, final, dates, start=None, end=None, 
                              strategyName="Grid Trading", showIndex=True, 
                              figuresize=(12, 6)):
    dates = pd.to_datetime(dates)
    
    # 时间筛选
    if start is not None or end is not None:
        start_ts = pd.Timestamp(start) if start else dates.min()
        end_ts   = pd.Timestamp(end)   if end   else dates.max()
        mask  = (dates >= start_ts) & (dates <= end_ts)
        res   = res[mask.values].reset_index(drop=True)
        dates = dates[mask.values]

    fig, ax1 = plt.subplots(figsize=figuresize)
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
                          "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10})

    # -------- Equity + Cash --------
    ax1.plot(dates, res["equity"], color="tab:blue",  label="Equity")
    ax1.plot(dates, res["cash"],   color="tab:green", linestyle="--", label="Cash")
    ax1.set_ylabel("Equity / Cash")
    ax1.tick_params(axis="y")
    ax1.grid(True)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    # -------- Position --------
    ax2 = ax1.twinx()
    ax2.plot(dates, res["position"], color="tab:orange", alpha=0.7, label="Position")
    ax2.set_ylabel("Position", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    if showIndex:
        ax1.set_title(f"{strategyName}: "
                      f"CAGR {stats['CAGR']:.2%}, "
                      f"Trades {final['trades']}, "
                      f"Fees {final['fees_paid']:.2f}, "
                      f"Sharpe {stats['Sharpe']:.2f}, "
                      f"MaxDD {stats['MaxDD']:.2%}, "
                      f"Calmar {stats['Calmar']:.2f}")

    fig.tight_layout()
    plt.show()