from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import numpy as np
import json

class Trader:
    def compute_volatility(self, traderData, vwap):
        window = 10  # Number of past VWAP prices to consider
        if "price_history" not in traderData:
            traderData["price_history"] = []
        
        traderData["price_history"].append(vwap)
        
        if len(traderData["price_history"]) > window:
            traderData["price_history"].pop(0)
        
        if len(traderData["price_history"]) > 1:
            returns = np.diff(traderData["price_history"])  # Compute price changes
            return np.std(returns)
        else:
            return 1  # Default to small volatility

    def compute_order_book_imbalance(self, order_depth):
        total_bid = sum(abs(size) for size in order_depth.buy_orders.values())
        total_ask = sum(abs(size) for size in order_depth.sell_orders.values())
        return (total_bid - total_ask) / (total_bid + total_ask) if (total_bid + total_ask) > 0 else 0

    def compute_price_trend(self, traderData):
        history = traderData.get("price_history", [])
        if len(history) < 5:
            return 0  # Not enough data
        return np.mean(np.diff(history[-5:]))  # Recent trend

    def get_vwap(self, buy_orders, sell_orders):
        total_size = 0
        weighted_price_sum = 0
        bid_prices = sorted(buy_orders, reverse=True)  # Best bid first
        ask_prices = sorted(sell_orders)  # Best ask first

        for price in bid_prices:
            size = buy_orders[price]
            weighted_price_sum += price * abs(size)
            total_size += abs(size)
        for price in ask_prices:
            size = sell_orders[price]
            weighted_price_sum += price * abs(size)
            total_size += abs(size)

        return weighted_price_sum / total_size if total_size else (bid_prices[0] + ask_prices[0]) / 2
    
    def run(self, state: TradingState):
        result = {}
        traderData = json.loads(state.traderData) if state.traderData else {}
        
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
            best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
            vwap = self.get_vwap(order_depth.buy_orders, order_depth.sell_orders)
            
            if product == 'RAINFOREST_RESIN':
                mid_price = 10000
                base_target_profit = 4
                spread = 8
            elif product == 'KELP':
                mid_price = vwap
                base_target_profit = 0.7
                spread = 1.5
            
            position = state.position.get(product, 0)
            max_position = 50
            available_long = max_position - position
            available_short = max_position + position
            
            # Compute adaptive factors
            volatility = self.compute_volatility(traderData, (best_ask+best_bid)/2)
            imbalance = self.compute_order_book_imbalance(order_depth)
            trend = self.compute_price_trend(traderData)
            
            # Adjust target profit based on volatility
            target_profit = base_target_profit * (1 + min(0.5, volatility))
            
            # Adjust fair price dynamically
            inventory_skew = -(position / max_position) * (1 + abs(position) / max_position) #* (spread / 2)
            fair_price = mid_price + (imbalance * (spread / 2)) + (trend * 0.5)
            
            # Adjust bid/ask based on adaptive pricing
            bid_price = round(fair_price - target_profit)
            ask_price = round(fair_price + target_profit)
            size_multiplier = 1 - abs(position) / max_position
            bid_size = int(min(available_long, 42) * size_multiplier)
            ask_size = int(min(available_short, 42) * size_multiplier)
            
            if bid_size > 0:
                if best_bid > bid_price and best_bid < fair_price:
                    orders.append(Order(product, best_bid+1, bid_size))
                else:
                    orders.append(Order(product, bid_price, bid_size))
            if ask_size > 0:
                if best_ask < ask_price and best_ask > fair_price:
                    orders.append(Order(product, best_ask-1, -ask_size))
                else:
                    orders.append(Order(product, ask_price, -ask_size))
            
            # Opportunistic trading
            available_long -= bid_size
            available_short -= ask_size
            if best_ask < mid_price and -best_ask_amount <= available_long:
                orders.append(Order(product, best_ask, -best_ask_amount))
            if best_bid > mid_price and best_bid_amount <= available_short:
                orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
        
        return result, None, json.dumps(traderData)
