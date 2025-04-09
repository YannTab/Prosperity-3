from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict


class Trader:

	def run(self, state: TradingState):

		def get_vwap(buy_orders, sell_orders):
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

		result = {}
		traderData = state.traderData  # Can be used to persist state across runs

		for product in state.order_depths:
			order_depth: OrderDepth = state.order_depths[product]
			orders: List[Order] = []
			best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
			best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
			mm_ask, _ = list(order_depth.sell_orders.items())[-1]
			mm_bid, _ = list(order_depth.buy_orders.items())[-1]
			vwap = get_vwap(order_depth.buy_orders, order_depth.sell_orders)
			print(f"VWAP: {vwap} ")

			if product == 'RAINFOREST_RESIN':
				mid_price = vwap
				target_profit = 4
				change_factor = 1
			elif product == 'KELP':
				mid_price = vwap
				target_profit = 0.7
				change_factor = 1
			position = state.position.get(product, 0)
			print(f'Position:{position} ')
			max_position = 50
			available_long = max_position - position
			available_short = max_position + position

			# Adjust fair price based on inventory
			
			long_inventory_skew = int((max(0, position) / max_position)/change_factor)
			short_inventory_skew = int((max(0, -position) / max_position)/change_factor)
			
			fair_price = mid_price 

			# Slightly below fair price
			bid_price = round(fair_price - target_profit) - long_inventory_skew
			# Slightly above fair price
			ask_price = round(fair_price + target_profit) + short_inventory_skew

			bid_size_multiplier = 1 - max(0, position) / max_position
			ask_size_multiplier = 1 - max(0, -position) / max_position


			
			
			if state.timestamp in [93200, 154100] and product == 'KELP':
				orders.append(Order(product, mm_ask, int(available_long)))
				print(f'Trend {product} - BUY {int(available_long)} - @ {mm_ask}')
				result[product] = orders	
				continue
			if state.timestamp in [34400, 119800] and product == 'KELP':
				orders.append(Order(product, mm_bid, -int(available_short)))
				print(f'Trend {product} - SELL {int(available_short)} - @ {mm_bid}')
				result[product] = orders
				continue

			if best_ask < mid_price and -best_ask_amount <= available_long:
				orders.append(Order(product, best_ask, -best_ask_amount))
				print(f'Arbitraged {product} - BUY  max {best_ask_amount} - @ {best_ask}')
				available_long = available_long + best_ask_amount				
			if best_bid > mid_price and best_bid_amount <= available_short:
				orders.append(Order(product, best_bid, -best_bid_amount))
				print(f'Arbitraged {product} - SELL  max {best_bid_amount} - @ {best_bid}')
				available_short = available_short + best_bid_amount

			bid_size = int(min(available_long,
							29)*bid_size_multiplier)  # Limit order size
			ask_size = int(min(available_short,
							29)*ask_size_multiplier)  # Limit order size

			if bid_size > 0:
				if best_bid > bid_price and best_bid < fair_price: # align ourselves
					orders.append(Order(product, best_bid+1, bid_size))
					print(f'Market Maked {product} - BUY {bid_size} - @ {best_bid+1}')
				else:
					orders.append(Order(product, bid_price, bid_size)) # normal quote
					print(f'Market Maked {product} - BUY {bid_size} - @ {bid_price}')
			if ask_size > 0:
				if best_ask < ask_price and best_ask > fair_price:
					orders.append(Order(product, best_ask-1, -ask_size))
					print(f'Market Maked {product} - SELL {ask_size} - @ {best_ask-1}')
				else:
					orders.append(Order(product, ask_price, -ask_size))
					print(f'Market Maked {product} - SELL {ask_size} - @ {ask_price}')	

			#### Arbitrage part ##### (Not really arbitrage, more like opportunistic trading)
			

			

			result[product] = orders

		return result, None, "UPDATED_STATE"
