"""
Trend-following live trader for IBKR paper/live account.
Polls every CHECK_SEC seconds; buys on uptrend, sells on trend reversal or SL.

python trend_follower.py
# 挂后台，关终端不停
nohup python trend_follower.py > /dev/null 2>&1 &
# 或者用 screen
screen -S trader
python trend_follower.py
# Ctrl+A D 退出 screen，进程继续跑
# screen -r trader  重新进入

Stop:
    Ctrl+C  (trade log saved to trade_log_YYYYMMDD_HHMMSS.csv)
    or kill
"""

import signal
import sys
import threading
import time
import logging
import os
from datetime import datetime

import pandas as pd
import numpy as np
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order

# ── Logging: console + file ────────────────────────────────────────────────────
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(LOG_DIR, f'trend_follower_{datetime.now().strftime("%Y%m%d")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding='utf-8'),
    ]
)
log = logging.getLogger('TrendFollower')

# ── Config ─────────────────────────────────────────────────────────────────────
SYMBOL    = 'NVDA'
BAR_SIZE  = '30 mins'
DURATION  = '30 D'
TREND_WIN = 42        # bars in trend window  (42 × 30min ≈ 3.2 trading days)
TREND_LB  = 65        # bars for MA lookback  (65 × 30min ≈ 5 trading days)
TREND_THR = 0.0005    # normalized slope threshold per bar
SHARES    = 10
SL_PCT    = 0.03
PORT      = 7497      # 7497=paper, 7496=live
CLIENT_ID = 11
CHECK_SEC = 1800      # poll interval — 30 min

# 考虑基本面、价格和换手率

# ── Trend detection ────────────────────────────────────────────────────────────
def compute_price_trend(close_arr, win=TREND_WIN, lb=TREND_LB, thr=TREND_THR):
    """
    Multi-method normalized slope → 'uptrend' | 'fluctuation' | 'downtrend' | 'unknown'
    M1: linear fit on last win bars
    M2: linear fit on win-bar MA over lb bars
    M3: linear fit on 6-bar MA over win bars
    M4: exponentially weighted linear fit
    """
    n = len(close_arr)
    if n < lb + 4:
        return 'unknown'

    LAMBDA = 0.8
    xwin = np.arange(win, dtype=float)
    wexp = np.array([LAMBDA ** (win - 1 - i) for i in range(win)])

    def _median_slope(end_idx):
        p_win = np.array(close_arr[end_idx - win + 1: end_idx + 1], dtype=float)
        p_lb  = np.array(close_arr[end_idx - lb  + 1: end_idx + 1], dtype=float)
        pmean = np.mean(p_win)
        if pmean == 0:
            return None
        sl = []

        s1, _ = np.polyfit(xwin, p_win, 1)
        sl.append(s1 / pmean)

        ma = np.convolve(p_lb, np.ones(win) / win, mode='valid')
        if len(ma) >= 3:
            s2, _ = np.polyfit(np.arange(len(ma), dtype=float), ma, 1)
            sl.append(s2 / np.mean(ma))

        ma6 = np.convolve(p_win, np.ones(6) / 6, mode='valid')
        if len(ma6) >= 3:
            s3, _ = np.polyfit(np.arange(len(ma6), dtype=float), ma6, 1)
            sl.append(s3 / np.mean(ma6))

        s4, _ = np.polyfit(xwin, p_win, 1, w=wexp)
        sl.append(s4 / pmean)

        return float(np.median(sl))

    med = _median_slope(n - 1)
    if med is None:
        return 'unknown'
    if med > thr:
        return 'uptrend'
    elif med < -thr:
        return 'downtrend'
    else:
        return 'fluctuation'


# ── IBKR client ────────────────────────────────────────────────────────────────
class IBTrendFollower(EWrapper, EClient):

    def __init__(self):
        EClient.__init__(self, self)
        self.pos         = 0
        self.avg_cost    = 0.0
        self.entry_price = None
        self.prev_trend  = 'unknown'
        self.next_oid    = 1
        self.trade_log   = []
        self._bars       = []
        self._connected  = threading.Event()
        self._hist_done  = threading.Event()
        self._oid_ready  = threading.Event()
        self._stop       = threading.Event()

    # ── EWrapper callbacks ────────────────────────────────────────────────────
    def nextValidId(self, orderId):
        self.next_oid = orderId
        self._connected.set()
        self._oid_ready.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=''):
        if errorCode not in (2104, 2106, 2107, 2108, 2158):
            log.warning(f'[IBKR err {errorCode}] {errorString}')

    def historicalData(self, reqId, bar):
        self._bars.append({
            'date': bar.date, 'open': bar.open,
            'high': bar.high, 'low': bar.low,
            'close': bar.close, 'volume': bar.volume,
        })

    def historicalDataEnd(self, reqId, start, end):
        self._hist_done.set()

    def position(self, account, contract, pos, avgCost):
        if contract.symbol == SYMBOL:
            self.pos = int(pos)
            self.avg_cost = float(avgCost)
            log.info(f'Position sync: {SYMBOL} {self.pos} shares @ avg {avgCost:.2f}')

    def positionEnd(self):
        pass

    def orderStatus(self, orderId, status, filled, remaining,
                    avgFillPrice, permId, parentId, lastFillPrice,
                    clientId, whyHeld, mktCapPrice):
        if filled > 0:
            log.info(f'Order {orderId} {status}: filled={filled} @ {avgFillPrice:.2f}')

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _contract(self):
        c = Contract()
        c.symbol = SYMBOL
        c.secType = 'STK'
        c.exchange = 'SMART'
        c.currency = 'USD'
        return c

    def _next_oid(self):
        self._oid_ready.clear()
        self.reqIds(-1)
        self._oid_ready.wait(timeout=5)
        oid = self.next_oid
        self.next_oid += 1
        return oid

    def fetch_bars(self):
        self._bars = []
        self._hist_done.clear()
        self.reqHistoricalData(
            reqId=1, contract=self._contract(), endDateTime='',
            durationStr=DURATION, barSizeSetting=BAR_SIZE,
            whatToShow='TRADES', useRTH=1, formatDate=1,
            keepUpToDate=False, chartOptions=[],
        )
        if not self._hist_done.wait(timeout=60):
            log.error('fetch_bars timeout')
        df = pd.DataFrame(self._bars)
        if df.empty:
            return df
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        for col in ('open', 'high', 'low', 'close'):
            df[col] = df[col].astype(float)
        return df

    def _place_order(self, action, qty):
        o = Order()
        o.action = action
        o.orderType = 'MKT'
        o.totalQuantity = qty
        o.tif = 'DAY'
        oid = self._next_oid()
        self.placeOrder(oid, self._contract(), o)
        return oid

    def buy(self, price_ref):
        oid = self._place_order('BUY', SHARES)
        self.entry_price = price_ref
        log.info(f'BUY  oid={oid}  {SHARES} {SYMBOL} @ ~{price_ref:.2f}')
        self.trade_log.append({
            'time': pd.Timestamp.now(), 'action': 'BUY',
            'price': price_ref, 'shares': SHARES,
        })

    def sell(self, price_ref, reason):
        if self.pos <= 0:
            return
        qty = self.pos
        oid = self._place_order('SELL', qty)
        pnl = (price_ref - self.entry_price) * qty if self.entry_price else None
        log.info(f'SELL oid={oid}  {qty} {SYMBOL} @ ~{price_ref:.2f}  reason={reason}  pnl≈{pnl:.1f}')
        self.trade_log.append({
            'time': pd.Timestamp.now(), 'action': 'SELL',
            'price': price_ref, 'reason': reason,
            'shares': qty, 'pnl': pnl,
        })
        self.entry_price = None

    # ── Strategy logic ────────────────────────────────────────────────────────
    def check_and_trade(self):
        df = self.fetch_bars()
        if df.empty or len(df) < TREND_LB + 4:
            log.warning(f'Insufficient bars: {len(df)}')
            return None

        cur_trend  = compute_price_trend(df['close'].values)
        last_close = float(df['close'].iloc[-1])
        last_time  = df['date'].iloc[-1]

        log.info(
            f'[{last_time}]  close={last_close:.2f}  '
            f'trend={cur_trend}  prev={self.prev_trend}  pos={self.pos}'
        )

        if self.prev_trend != 'uptrend' and cur_trend == 'uptrend' and self.pos == 0:
            self.buy(last_close)
        elif self.prev_trend == 'uptrend' and cur_trend != 'uptrend' and self.pos > 0:
            self.sell(last_close, reason=f'trend->{cur_trend}')
        elif self.pos > 0 and self.entry_price:
            ret = (last_close - self.entry_price) / self.entry_price
            if ret <= -SL_PCT:
                self.sell(last_close, reason=f'SL {ret:.1%}')

        self.prev_trend = cur_trend

    # ── Connect / loop / stop ─────────────────────────────────────────────────
    def start(self):
        self.connect('127.0.0.1', PORT, clientId=CLIENT_ID)
        threading.Thread(target=self.run, daemon=True, name='IBKRMsg').start()
        if not self._connected.wait(timeout=10):
            raise RuntimeError(f'Connection timeout — is TWS/Gateway running on port {PORT}?')
        self.reqPositions()
        time.sleep(1)
        log.info(f'Connected  port={PORT}  clientId={CLIENT_ID}')

    def run_loop(self):
        """Blocking loop — runs in main thread. Returns when stop() is called."""
        log.info(f'Live loop started (interval={CHECK_SEC}s). Press Ctrl+C to stop.')
        while not self._stop.is_set():
            try:
                self.check_and_trade()
            except Exception as e:
                log.error(f'Loop error: {e}')
            self._stop.wait(CHECK_SEC)

    def stop(self):
        self._stop.set()
        self.disconnect()
        log.info('Stopped.')
        self._save_trade_log()

    def _save_trade_log(self):
        if not self.trade_log:
            log.info('No trades to save.')
            return
        fname = os.path.join(
            LOG_DIR, f'trade_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        pd.DataFrame(self.trade_log).to_csv(fname, index=False)
        log.info(f'Trade log saved to {fname}')


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    trader = IBTrendFollower()

    def _handle_signal(sig, frame):
        log.info('Interrupt received, shutting down...')
        trader.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    trader.start()
    trader.run_loop()   # blocks until Ctrl+C / SIGTERM


if __name__ == '__main__':
    main()
