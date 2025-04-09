from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
	
	def run(self, state: TradingState):

		def close(product, position, price, orders):
			if position > 0:
				# Sell if we are long
				orders.append(Order(product, price, -position))
			else:
				# Buy back if we are short
				orders.append(Order(product, price, -position))

		# Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
		print("traderData: " + state.traderData)
		print("Observations: " + str(state.observations))
		result = {}
		for product in state.order_depths:
			order_depth: OrderDepth = state.order_depths[product]
			orders: List[Order] = []
			if product == 'RAINFOREST_RESIN':
				position = state.position.get(product, 0)
				max_position = 50
				acceptable_price = int(1e4);  # Participant should calculate this value
				print("Acceptable price : " + str(acceptable_price))
				print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

				available_long = max_position - position
				available_short = max_position + position

				if len(order_depth.sell_orders) != 0:
					best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
					if best_ask < acceptable_price:
						print("BUY", str(-best_ask_amount) + "x", best_ask)
						orders.append(Order(product, best_ask+1, available_long))
					elif best_ask == acceptable_price:
						print("Closed positions", best_ask)
						close(product, position, acceptable_price, orders)
					else:
						orders.append(Order(product, best_ask-1, int(-available_short*0.5)))
		
				if len(order_depth.buy_orders) != 0:
					best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
					if best_bid > acceptable_price:
						print("SELL", str(best_bid_amount) + "x", best_bid)
						orders.append(Order(product, best_bid-1, -available_short))
					elif best_bid == acceptable_price:
						print("Closed positions", best_ask)
						close(product, position, acceptable_price, orders)
					else:
						orders.append(Order(product, best_bid+1, int(available_long*0.5)))
				
				result[product] = orders
	
	
		traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
		
		return result, None, traderData
