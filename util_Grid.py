import numpy as np
import pandas as pd


# ========================== buy / sell partial ========================== 
def buy_partial(state: dict, price: float, size: float, row_time: pd.Timestamp, fee: float = 0.0) -> None:
    if size <= 0:
        return

    cost = price * size
    if cost > state["cash"]:
        size = state["cash"] / price
        cost = price * size
        if size <= 0:
            return

    trade_fee = cost * fee

    # update avg price
    old_pos = state["position"]
    new_pos = old_pos + size

    if old_pos <= 0:
        state["avg_price"] = price
    else:
        state["avg_price"] = (state["avg_price"] * old_pos + price * size) / new_pos

    state["cash"] -= (cost + trade_fee)
    state["position"] = new_pos
    state["trades"] += 1
    state["fees_paid"] += trade_fee


def sell_partial(state: dict, price: float, size: float, row_time: pd.Timestamp, fee: float = 0.0) -> None:
    if size <= 0:
        return
    if state["position"] <= 0:
        return

    size = min(size, state["position"])
    proceeds = price * size
    trade_fee = proceeds * fee

    state["cash"] += (proceeds - trade_fee)
    state["position"] -= size

    if state["position"] <= 1e-12:
        state["position"] = 0.0
        state["avg_price"] = np.nan

    state["trades"] += 1
    state["fees_paid"] += trade_fee
    
    
# ========================== add tradeLog ========================== 
def add_tradelog(trade, ttype: str, layer:int, price: float, size:float, time:pd.Timestamp, trade_time:int,
                 fee:float, position:int, anchor:float, next_buy:float, next_sell:float):
    ''' time: yyyy-mm-dd
        trade_time - sell/buy times in one day
        position - after trade
    '''
    trade.append({"time": pd.to_datetime(time), "trade_time":trade_time, "anchor":anchor,
                  "type": ttype, "layer":layer, 
                    "price": price, "size": size, "fee": price*size*fee,
                    "next_buy": next_buy, "next_sell": next_sell,
                    "position_after": position})
    


# ========================== Strategy GridTrading ========================== 
from collections import deque

class Strategy_GridTrading:
    """
    Long-only Grid Trading (mean-reversion style), bar-by-bar.

    Concept:
      - Use a rolling center price (e.g., SMA) as the grid anchor.
      - When price drops by grid_step below the current buy level -> buy one layer.
      - When price rises by grid_step above the current sell level -> sell one layer.
      - Risk control: if price falls too far below anchor -> exit all (stop).
    ----------
    center_n : int
        Rolling window for grid center (SMA). Larger = slower-moving anchor.
    grid_step : float
        Grid spacing in *percentage*, e.g. 0.01 means 1% per grid.
    max_layers : int
        Maximum number of buy layers (scale-ins) allowed.
    stop_pct : float
        Hard stop relative to anchor, e.g. 0.12 means -12% from anchor -> liquidate
    warmup : int | None
        Bars to skip trading (indicator warmup). Default = center_n + 2
    """

    def __init__(self, center_n=20, grid_step=0.01, max_layers=6, stop_pct=0.12, maxPositionDays=30, floor_position=0, warmup=None, 
                 reanchor_window=30, reanchor_threshold=0.25):
        self.center_n = int(center_n)
        self.grid_step = float(grid_step)
        self.max_layers = int(max_layers)
        self.stop_pct = float(stop_pct)
        self.maxPositionDays = int(maxPositionDays)
        self.floor_position = int(floor_position)
        self.warmup = warmup if warmup is not None else self.center_n + 2
        self.reanchor_window = reanchor_window
        self.reanchor_threshold = reanchor_threshold
        self._center_history = deque(maxlen=reanchor_window)  # 存最近 N 个 center 值

        # internal (per run)
        self.anchor = np.nan         # grid anchor (center at time of (re)start)
        self.layer = 0               # how many layers currently bought (0..max_layers)
        self.layer_sizes = {}        # keep sell/buy consistent, clear when reset
        self.next_buy = np.nan
        self.next_sell = np.nan

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["center"] = out["close"].rolling(self.center_n).mean()
        # --- ATR (True Range) ---
        prev_close = out["close"].shift()
        tr = pd.concat([
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"]  - prev_close).abs()
        ], axis=1).max(axis=1)
        tr.iloc[0] = out["high"].iloc[0] - out["low"].iloc[0]  # 第一行无前收盘，用 HL 代替

        out["atr"] = tr.rolling(14).mean()
        # --- ATR-adaptive grid_step ---
        out["grid_step"] = (out["atr"] / out["close"]).clip(0.005, 0.03)
        out["grid_step"] = out["grid_step"].bfill()

        return out

    def _reset_grid(self, anchor: float, layer: int, reset_layer_size=False) -> None:
        self.anchor = float(anchor)        
        self.layer = layer
        self.next_buy = self.anchor * (1.0 - self.grid_step)    # 0.00005 
        self.next_sell = self.anchor * (1.0 + self.grid_step)
        if reset_layer_size: self.layer_sizes = {}


    # reanchor when current price exceed threshold
    def _check_reanchor(self, c, row_time=pd.Timestamp, layer=0, position=0, trade_log=None):  
        deviation = (np.array(self._center_history) - self.anchor) / self.anchor
        if all(d > self.reanchor_threshold for d in abs(deviation)):
            old_anchor = self.anchor
            new_anchor = min(np.nanmean(np.array(self._center_history)), 9.06) #0.5*(old_anchor + np.nanmean(np.array(self._center_history)))
            self._reset_grid(anchor=new_anchor, layer=layer)
            add_tradelog(trade_log, layer=self.layer, ttype=f"[RESET ANCHOR]", price=c, size=0, 
                                 time=row_time, trade_time=999, fee=0, position=position,
                                anchor=self.anchor, next_buy=self.next_buy, next_sell=self.next_sell)

    @staticmethod
    def _cal_next_buy_price(_anchor, _layer, _step):
        _step = _step * (0.9 ** _layer)   #step adjust according to layer, prevent price too high/low
        return _anchor * (1.0 - (_layer+1) * _step)
    @staticmethod
    def _cal_next_sell_price(_anchor, _layer, _step):
        _step = _step * (0.9 ** _layer)  # step adjust 
        return _anchor * (1.0 + _layer * _step) 
    

    def on_bar(self, i: int, row_time: pd.Timestamp, row: pd.Series, 
               state: dict, fee: float, trade_log) -> None:
        c = float(row["close"])  # current price
        center = row["center"]
        step = row["grid_step"] if "grid_step" in row.index else self.grid_step

        if i < self.warmup or np.isnan(center):
            return

        # --- initiate ---
        if np.isnan(self.anchor):    # (first time)
            print(f'[Reset Grid]: 1st time, anchor={float(center):.4f}')
            self._reset_grid(anchor=center, layer=0, reset_layer_size=True)   #
            add_tradelog(trade_log, layer=self.layer, ttype="INITIATE", price=0, size=0, 
                            time=row_time, trade_time=0, fee=fee, position=0,
                            anchor=self.anchor, 
                            next_buy=self._cal_next_buy_price(self.anchor, self.layer, step),
                            next_sell=self._cal_next_sell_price(self.anchor, self.layer, step))

        in_pos = state["position"] > 0
        if (not in_pos) and (self.layer != 0):   # positoin>0 and layer==0  # why
            print('[Reset Grid]: position empty')
            self._reset_grid(anchor=float(center), layer=self.layer, reset_layer_size=True)
            

        # add condition, if price exceed threshold for reanchor_window days, reset
        self._center_history.append(center)
        self._check_reanchor(c, row_time=row_time, layer=self.layer, position=state["position"], trade_log=trade_log)

        # Hard stop (protect against one-way crash)
        # situation1: c << sell_price
        if in_pos and c <= (self.anchor * (1.0 - self.stop_pct)):
            print(f'[Hard stop] {row_time}, new anchor = {center}')
            sell_size = state["position"]
            sell_partial(state, price=c, size=sell_size, row_time=row_time, fee=fee)
            self._reset_grid(anchor=float(center), layer=0, reset_layer_size=True)  # restart grid after liquidation
            add_tradelog(trade_log, layer=self.layer, ttype="SELL[Hard Stop1]", price=c, size=sell_size, 
                            time=row_time, trade_time=1, fee=fee, position=state["position"],
                            anchor=self.anchor, 
                            next_buy=self._cal_next_buy_price(self.anchor, self.layer, step),
                            next_sell=self._cal_next_sell_price(self.anchor, self.layer, step))
            return
        
        # situation2: keep max position for long time (maxPositionDays)
        diff = pd.Timestamp(row_time) - pd.Timestamp(trade_log[-1]["time"])
        if self.layer == self.max_layers and diff.days > self.maxPositionDays:
            # 如果略高于平均成本价格：清仓。 别：如果现在价格低于买入的最低价格：清仓
            if c > (state["avg_price"]*(1+3*fee)):
                print(f'[Hard stop2] {row_time}, new anchor = {self.anchor}')
                sell_size = state["position"]
                sell_partial(state, price=c, size=sell_size, row_time=row_time, fee=fee)
                self._reset_grid(anchor=self.anchor, layer=0, reset_layer_size=True)  # restart grid after liquidation
                add_tradelog(trade_log, layer=self.layer, ttype="SELL[Hard Stop2]", price=c, size=sell_size, 
                                time=row_time, trade_time=1, fee=fee, position=state["position"],
                                anchor=self.anchor, 
                                next_buy=self._cal_next_buy_price(self.anchor, self.layer, step),
                                next_sell=self._cal_next_sell_price(self.anchor, self.layer, step))
                return
        

        # # --- buy method1 ---
        # if (self.layer < self.max_layers) and (c <= self.next_buy) and (state["cash"] > 0):
        #     # #--- size method0
        #     # buy_size = int(state["target_size"] / max(self.max_layers, 1))
        #     #--- size method1
        #     weight = (self.max_layers - self.layer + 1)
        #     buy_size = int(state["target_size"] * weight / (sum(range(1, self.max_layers + 2))))  ##
        #     # #--- buy size method2: simple linear
        #     # layer_size = state["target_size"] / max(self.max_layers, 1)
        #     # buy_size = int(layer_size * (1+0.1*self.layer))
        #     buy_partial(state, price=c, size=buy_size, row_time=row_time, fee=fee)
        #     self.layer += 1
        #     self.next_buy = self.anchor * (1.0 - self.layer * step)
        #     self.next_sell = self.anchor * (1.0 + step)   # always 1 step
        #     add_tradelog(trade_log, ttype="BUY", price=c, size=buy_size, time=row_time, fee=fee, position=state["position"],
        #                   next_buy=self.next_buy, next_sell=self.next_sell)

        # # --- sell method1 ---
        # if (self.layer > 0) and (c >= self.next_sell) and (state["position"] > 0):
        #     # # --- sell size method0
        #     # sell_size = int(state["target_size"] / max(self.max_layers, 1))    # int(layer_size* (self.layer+1)/max_)
        #     # --- sell size method1
        #     sell_size = int(state["target_size"] * (self.layer + 3) / (sum(range(1, self.max_layers + 2)))) ##
        #     sell_partial(state, price=c, size=sell_size, row_time=row_time, fee=fee)
        #     self.layer -= 1
        #     self.next_sell = self.anchor * (1.0 + self.layer * self.grid_step)
        #     self.next_buy  = self.anchor * (1.0 - (self.layer + 1) * self.grid_step)
        #     add_tradelog(trade_log, ttype="SELL", price=c, size=sell_size, time=row_time, fee=fee, position=state["position"],
        #                   next_buy=self.next_buy, next_sell=self.next_sell)
        #     if state["position"] <= 0:
        #         self._reset_grid(anchor=float(center))
        #     if self.layer == 0:  #
        #     #     print("layer+2 ---") 
        #     #     self.layer +=2
        #         add_tradelog(trade_log, ttype="LAYER==0", price=c, size=999, time=row_time, fee=fee, position=state["position"],
        #                     next_buy=self.next_buy, next_sell=self.next_sell)

        _denom = (self.max_layers + 1) * (self.max_layers + 2) // 2

        # --- buy method2 ---
        if (self.layer < self.max_layers):    # self.next_price循环
            self.next_buy = self._cal_next_buy_price(self.anchor, self.layer, step)#self.anchor * (1.0 - self.layer * step)
            self.next_sell = self._cal_next_sell_price(self.anchor, self.layer, step)

            trade_time = 1
            while (c <= self.next_buy) and (state["cash"] > 0) and (self.layer < self.max_layers):
                self.layer += 1
                # buy_size = int(state["target_size"] / max(self.max_layers, 1))
                weight = (self.max_layers - self.layer + 1)
                buy_size = int(state["target_size"] * weight / _denom)  ##
                self.layer_sizes[self.layer] = buy_size
                # #--- buy size method2: simple linear
                # layer_size = state["target_size"] / max(self.max_layers, 1)
                # buy_size = int(layer_size * (1+0.1*self.layer))

                # buy
                buy_partial(state, price=c, size=buy_size, row_time=row_time, fee=fee)
                add_tradelog(trade_log, layer=self.layer, ttype="BUY", price=c, size=buy_size, 
                             time=row_time, trade_time=trade_time, fee=fee, position=state["position"],
                            anchor=self.anchor, next_buy=self.next_buy, next_sell=self.next_sell)
                trade_time+=1

                self.next_buy = self._cal_next_buy_price(self.anchor, self.layer, step) #self.anchor * (1.0 - self.layer * step)
                self.next_sell = self._cal_next_sell_price(self.anchor, self.layer, step)
                if self.layer == self.max_layers:  #
                    print(f"max layer: {row_time} ---") 
                    add_tradelog(trade_log, layer=self.layer, ttype=f"[BUY]LAYER=={self.max_layers}", price=c, size=0, 
                                 time=row_time, trade_time=999, fee=fee, position=state["position"],
                                anchor=self.anchor, next_buy=self.next_buy, next_sell=self.next_sell)



        # --- sell method2 ---
        if (self.layer > 0):    # self.next_price循环
            self.next_sell = self._cal_next_sell_price(self.anchor, self.layer, step)
            self.next_buy  = self._cal_next_buy_price(self.anchor, self.layer, step)
            
            trade_time = 1
            while (c >= self.next_sell) and (state["position"] > 0) and (self.layer > 0):
                # sell_size = int(state["target_size"] / max(self.max_layers, 1))    # int(layer_size* (self.layer+1)/max_)
                exact_size = self.layer_sizes[self.layer]
                sell_size = exact_size#min(int(state["target_size"] * (self.layer + 1) / _denom), exact_size) ##

                # sell
                sell_partial(state, price=c, size=sell_size, row_time=row_time, fee=fee)
                add_tradelog(trade_log, layer=self.layer, ttype="SELL", price=c, size=sell_size, 
                             time=row_time, trade_time=trade_time, fee=fee, position=state["position"],
                            anchor=self.anchor, next_buy=self.next_buy, next_sell=self.next_sell)
                self.layer -= 1
                trade_time+=1

                if state["position"] <= self.floor_position:   # 留仓底干啥
                    print(f"[Reset Grid]: {row_time} position <= floor position")
                    self._reset_grid(anchor=min(float(center),9.06), layer=self.layer)
                    break  # avoid do while
                
                self.next_sell = self._cal_next_sell_price(self.anchor, self.layer, step)
                self.next_buy  = self._cal_next_buy_price(self.anchor, self.layer, step)

                if self.layer == 0: 
                    print(f"[SELL] layer==0: {row_time} ---") 
                    add_tradelog(trade_log, layer=self.layer, ttype="[SELL]LAYER==0", price=c, size=0, 
                                 time=row_time, trade_time=999, fee=fee, position=state["position"],
                                anchor=self.anchor, next_buy=self.next_buy, next_sell=self.next_sell)
                    # if state["position"]>0:   # clear position
                    #     sell_size = state["position"]
                    #     sell_partial(state, price=c, size=sell_size, row_time=row_time, fee=fee)
                    #     add_tradelog(trade_log, layer=self.layer, ttype="SELL", price=c, size=sell_size, 
                    #                 time=row_time, trade_time=trade_time, fee=fee, position=state["position"],
                    #                 anchor=self.anchor, next_buy=self.next_buy, next_sell=self.next_sell)
                    
                    # print(f'[Reset Grid]: layer=0 after sell, anchor={c}')
                    # self._reset_grid(anchor=c)  # average price for previous 10 days
