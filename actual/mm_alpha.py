from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any
import json
import numpy as np
import pandas as pd
import copy

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

logger = Logger()

class Product():
    def __init__(self, product, position, max_position, target_profit, order_depth, fair_price=None, quote_base_volume=1):
        self.position = position
        self.max_position = max_position
        self.target_profit = target_profit
        self.quote_base_volume = quote_base_volume
        self.product = product
        self.order_depth: OrderDepth = copy.deepcopy(order_depth)
        self.fair_price = fair_price if fair_price else self.get_vwap(order_depth.buy_orders, order_depth.sell_orders)
        self.best_ask, self.best_ask_amount = list(self.order_depth.sell_orders.items())[0]
        self.best_bid, self.best_bid_amount = list(self.order_depth.buy_orders.items())[0]
        self.mm_ask, _ = list(self.order_depth.sell_orders.items())[-1]
        self.mm_bid, _ = list(self.order_depth.buy_orders.items())[-1]
        logger.print(f"Fair: {fair_price} ")
        self.available_long = self.max_position - self.position
        self.available_short = self.max_position + self.position
        self.bid_price = round(self.fair_price - self.target_profit)
        self.ask_price = round(self.fair_price + self.target_profit)

    def get_vwap(self, buy_orders, sell_orders):
            # Compute Global Liquidity-Weighted Average Price (LWAP)
            total_size = 0
            weighted_price_sum = 0
            bid_prices = sorted(buy_orders, reverse=True)  # Best bid first
            ask_prices = sorted(sell_orders)  # Best ask first
            for price in bid_prices[-1:]:
                size = buy_orders[price]
                weighted_price_sum += price * abs(size)
                total_size += abs(size)
            for price in ask_prices[-1:]:
                size = sell_orders[price]
                weighted_price_sum += price * abs(size)
                total_size += abs(size)

            vwap = weighted_price_sum / total_size if total_size else (bid_prices[0] + ask_prices[0]) / 2
            return vwap

    def ema(self, prices, period):
        if len(prices) < period+5:
            return None, None # Not enough data to compute EMA
        # weights = np.exp(np.linspace(-1., 0., period))
        # weights /= weights.sum()
        # return np.convolve(prices, weights, mode='valid')[-1]
        ewm_ema = pd.Series(prices).rolling(period).mean()
        return ewm_ema.iloc[-2], ewm_ema.iloc[-1]
    
    def volatility(self, prices, period=100):
        if len(prices) < period:
            return None  # Not enough data to compute volatility
        #return np.std(prices[-period:])
        return pd.Series(prices).rolling(period).std().iloc[-1]

    def market_make(self, orders):
        bid_size = int(min(self.available_long, self.quote_base_volume))  # Limit order size
        ask_size = int(min(self.available_short, self.quote_base_volume))  # Limit order size

        if bid_size > 0:
            if self.best_bid and self.best_bid >= self.bid_price and self.best_bid < round(self.fair_price)-1: # align ourselves
                if abs(self.best_bid_amount) <= 1:
                    orders.append(Order(self.product, self.best_bid, bid_size))
                    logger.print(f'Market Maked {self.product} - BUY {bid_size} - @ {self.best_bid}')
                else:
                    orders.append(Order(self.product, self.best_bid+1, bid_size))
                    logger.print(f'Market Maked {self.product} - BUY {bid_size} - @ {self.best_bid+1}')
            else:
                orders.append(Order(self.product, self.bid_price, bid_size)) # normal quote
                logger.print(f'Market Maked {self.product} - BUY {bid_size} - @ {self.bid_price}')

        if ask_size > 0:
            if self.best_ask and self.best_ask <= self.ask_price and self.best_ask > round(self.fair_price)+1:
                if abs(self.best_ask_amount) <= 1:
                    orders.append(Order(self.product, self.best_ask, -ask_size))
                    logger.print(f'Market Maked {self.product} - SELL {ask_size} - @ {self.best_ask}')
                else:
                    orders.append(Order(self.product, self.best_ask-1, -ask_size))
                    logger.print(f'Market Maked {self.product} - SELL {ask_size} - @ {self.best_ask-1}')
            else:
                orders.append(Order(self.product, self.ask_price, -ask_size))
                logger.print(f'Market Maked {self.product} - SELL {ask_size} - @ {self.ask_price}')

    def market_take(self, orders):
        if self.best_ask < self.fair_price and -self.best_ask_amount <= self.available_long:
            # Take +ev trades
            orders.append(Order(self.product, self.best_ask, -self.best_ask_amount))
            logger.print(f'Arbitraged {self.product} - BUY  max {-self.best_ask_amount} - @ {self.best_ask}')
            self.available_long = self.available_long + self.best_ask_amount
            # Compute new bid prices  and bid size after arbitrage
            self.order_depth.sell_orders.pop(self.best_ask)
            if self.order_depth.sell_orders:
                self.best_ask, self.best_ask_amount = list(self.order_depth.sell_orders.items())[0]
            else:
                self.best_ask, self.best_ask_amount = None, None
        if self.best_bid > self.fair_price and self.best_bid_amount <= self.available_short:
            # Take + ev trade
            orders.append(Order(self.product, self.best_bid, -self.best_bid_amount))
            logger.print(f'Arbitraged {self.product} - SELL  max {self.best_bid_amount} - @ {self.best_bid}')
            self.available_short = self.available_short - self.best_bid_amount
            # Compute new ask prices after arbitrage
            self.order_depth.buy_orders.pop(self.best_bid)
            if self.order_depth.buy_orders:
                self.best_bid, self.best_bid_amount = list(self.order_depth.buy_orders.items())[0]
            else:
                self.best_bid, self.best_bid_amount = None, None

    def market_settle(self, orders):
        
        if self.best_bid == self.fair_price and self.position > 0:
            # Close long positions at fair price
            orders.append(Order(self.product, self.best_bid, -self.best_bid_amount))#max(-best_bid_amount, -position)))
            logger.print(f'Closed {self.product} - SELL {self.best_bid_amount} - @ {self.best_bid}')
            self.available_short = self.available_short - self.best_bid_amount
            # Compute new ask prices after position exit
            self.order_depth.buy_orders.pop(self.best_bid)
            if self.order_depth.buy_orders:
                self.best_bid, self.best_bid_amount = list(self.order_depth.buy_orders.items())[0]
            else:
                self.best_bid, self.best_bid_amount = None, None
        if self.best_ask == self.fair_price and self.position < 0:
            # Close short positions at fair price
            orders.append(Order(self.product, self.best_ask, -self.best_ask_amount))
            logger.print(f'Closed {self.product} - BUY {-self.best_ask_amount} - @ {self.best_ask}')
            self.available_long = self.available_long + self.best_ask_amount
            # Compute new bid prices after position exit
            self.order_depth.sell_orders.pop(self.best_ask)
            if self.order_depth.sell_orders:
                self.best_ask, self.best_ask_amount = list(self.order_depth.sell_orders.items())[0]
            else:
                self.best_ask, self.best_ask_amount = None, None
    
    def get_safe_trade_size(self, signal, desired_size, current_position):
        if signal == "BUY":
            max_size = self.available_long - max(current_position, 0)
            return int(min(desired_size, max_size))
        elif signal == "SELL":
            max_size = self.available_short + min(current_position, 0)
            return int(min(desired_size, max_size))
        return 0


class Kelp(Product):
    def __init__(self, product, position, max_position, target_profit, order_depth, fair_price=None, quote_base_volume=1):
        super().__init__(product, position, max_position, target_profit, order_depth, fair_price, quote_base_volume)
        self.kelp_volatility_mean = 0.9
        self.signal_expiration = 1

    def signal_sma_alpha(self, traderData, orders, result):
        # Store price history
        if self.product not in traderData:
            traderData[self.product] = {'prices':[], 
                                    'signal':None,
                                    'signal_age':0}
        traderData[self.product]['prices'].append(self.fair_price)
    

        if len(traderData[self.product]['prices']) > 500:
            traderData[self.product]['prices'].pop(0)

        # Compute indicators
        price_history = traderData[self.product]['prices']
        signal_age = traderData[self.product]['signal_age']
        SIGNAL = traderData[self.product]['signal']
        ema50_prev, ema50_now = self.ema(price_history, 100)
        ema100_prev, ema100_now = self.ema(price_history, 300)
        vol100 = self.volatility(price_history, 100)

        if ema50_prev is not None and ema100_prev is not None and ema50_now is not None and ema100_now and vol100 is not None:
            if ema50_prev < ema100_prev and ema50_now > ema100_now and vol100 > self.kelp_volatility_mean:
                if SIGNAL is None:
                    SIGNAL = "BUY"
                    signal_age = 0
            elif ema50_prev > ema100_prev and ema50_now < ema100_now and vol100 > self.kelp_volatility_mean:
                if SIGNAL is None:
                    SIGNAL = "SELL"
                    signal_age = 0

        # Manage signal expiration
        if SIGNAL is not None:
            signal_age += 1
            if signal_age > self.signal_expiration:
                SIGNAL = None
                signal_age = 0
        
        traderData[self.product]['signal'] = SIGNAL
        traderData[self.product]['signal_age'] = signal_age

        logger.print(f'Volatility:{vol100}, signal:{SIGNAL}, age:{signal_age}')

        if SIGNAL == "BUY" :
            orders.append(Order(self.product, self.mm_ask, int(self.available_long)))
            logger.print(f'Trend {self.product} - BUY {int(self.available_long)} - @ {self.mm_ask}')
            traderData[self.product]['long_alpha_target'] = 0
            result[self.product] = orders
        if SIGNAL == "SELL":
            orders.append(Order(self.product, self.mm_bid, -int(self.available_short)))
            logger.print(f'Trend {self.product} - SELL {int(self.available_short)} - @ {self.mm_bid}')
            traderData[self.product]['short_alpha_target'] = 0
            result[self.product] = orders
        return SIGNAL
    

class Rainforest_Resin(Product):
    def __init__(self, product, position, max_position, target_profit, order_depth, fair_price=None, quote_base_volume=1):
        super().__init__(product, position, max_position, target_profit, order_depth, fair_price, quote_base_volume)


class Squid_Ink(Product):
    def __init__(self, product, position, max_position, target_profit, order_depth, fair_price=None, quote_base_volume=1):
        super().__init__(product, position, max_position, target_profit, order_depth, fair_price, quote_base_volume)
        self.signal_expiration = 4

    def signal_extrema_reversion(self, traderData, orders, result):
        if self.product not in traderData:
            traderData[self.product] = {
                'prices': [],
                'returns': [],
            }
            
        traderData[self.product]['prices'].append(self.fair_price)

        if len(traderData[self.product]['prices']) > 50:
            traderData[self.product]['prices'].pop(0)

        # Compute return
        prices = traderData[self.product]['prices']
        if len(prices) >= 2:
            ret = prices[-1] / prices[-2] - 1
            traderData[self.product]['returns'].append(ret)
        if len(traderData[self.product]['returns']) > 50:
            traderData[self.product]['returns'].pop(0)

        returns = traderData[self.product]['returns']
        position = self.position
        logger.print(f'Returns: {returns}')

        # Parameters
        up_threshold = 0.001326
        down_threshold = -0.0014

        # Entry logic
        if position == 0 and len(returns) >= 2:
            r_t_minus1 = returns[-2]
            r_t = returns[-1]

            if r_t < r_t_minus1 and r_t < down_threshold:
                # Enter long
                orders.append(Order(self.product, self.mm_ask, int(self.available_long)))
                logger.print(f'Extrema {self.product} - BUY {int(self.available_long)} - @ {self.mm_ask}')
                result[self.product] = orders

            elif r_t > r_t_minus1 and r_t > up_threshold:
                # Enter short
                orders.append(Order(self.product, self.mm_bid, -int(self.available_short)))
                logger.print(f'Extrema {self.product} - SELL {int(self.available_short)} - @ {self.mm_bid}')
                result[self.product] = orders

        # Exit logic — close position on next tick
        elif position != 0:
            exit_volume = int(self.available_long if position < 0 else self.available_short)
            exit_price = self.mm_ask if position < 0 else self.mm_bid
            exit_qty = exit_volume if position < 0 else -exit_volume

            orders.append(Order(self.product, exit_price, exit_qty))
            logger.print(f'Extrema {self.product} - EXIT {"SHORT" if position < 0 else "LONG"} {abs(exit_qty)} @ {exit_price}')
            result[self.product] = orders

        return 


class Trader:
    def __init__(self):
        pass
        
    def run(self, state: TradingState):
        result = {}
        traderData = json.loads(state.traderData) if state.traderData else {}  # Can be used to persist state across runs

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)
            logger.print(f'Position:{position}')

            if product == 'RAINFOREST_RESIN':
                resin_instance = Rainforest_Resin(
                    product='RAINFOREST_RESIN',
                    position=position,
                    max_position=50,
                    target_profit=8, 
                    order_depth=order_depth,
                    fair_price=10000,
                    quote_base_volume=38
                )
                resin_instance.market_take(orders=orders)
                resin_instance.market_settle(orders=orders)
                resin_instance.market_make(orders=orders)
                result[product] = orders

            if product == 'KELP':
                kelp_instance = Kelp(
                    product='KELP',
                    position=position,
                    max_position=50,
                    target_profit=0.8, 
                    order_depth=order_depth,
                    fair_price=None,
                    quote_base_volume=40
                )
                SIGNAL = kelp_instance.signal_sma_alpha(traderData=traderData, orders=orders, result=result)
                if SIGNAL:
                    continue
                kelp_instance.market_take(orders=orders)
                kelp_instance.market_make(orders=orders)
                result[product] = orders

            # if product == 'SQUID_INK':
            #     squid_ink_instance = Squid_Ink(
            #         product='SQUID_INK',
            #         position=position,
            #         max_position=25,
            #         target_profit=0.4, 
            #         order_depth=order_depth,
            #         fair_price=None,
            #         quote_base_volume=5
            #     )
            #     squid_ink_instance.signal_extrema_reversion(traderData=traderData, orders=orders, result=result)
            #     # SIGNAL = squid_ink_instance.signal_rsi_reversion(traderData=traderData, orders=orders, result=result)
            #     # if SIGNAL:
            #     #     continue
            #     result[product] = orders

        serialized_trader_data = json.dumps(traderData)
        logger.flush(state, result, None, serialized_trader_data)
        return result, None, serialized_trader_data



