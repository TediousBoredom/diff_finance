"""
Market Environment: 模拟真实市场环境
支持订单簿、撮合机制、市场冲击等
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"
    CANCEL = "cancel"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """订单"""
    agent_id: int
    order_type: OrderType
    side: OrderSide
    price: float
    quantity: float
    timestamp: int
    order_id: Optional[int] = None


@dataclass
class Trade:
    """成交记录"""
    buyer_id: int
    seller_id: int
    price: float
    quantity: float
    timestamp: int


class OrderBook:
    """订单簿"""
    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        self.bids: Dict[float, List[Order]] = {}  # 买单：价格 -> 订单列表
        self.asks: Dict[float, List[Order]] = {}  # 卖单：价格 -> 订单列表
        self.order_id_counter = 0
        self.orders: Dict[int, Order] = {}  # 所有活跃订单
        
    def add_order(self, order: Order) -> int:
        """添加订单"""
        order.order_id = self.order_id_counter
        self.order_id_counter += 1
        self.orders[order.order_id] = order
        
        if order.side == OrderSide.BUY:
            if order.price not in self.bids:
                self.bids[order.price] = []
            self.bids[order.price].append(order)
        else:
            if order.price not in self.asks:
                self.asks[order.price] = []
            self.asks[order.price].append(order)
            
        return order.order_id
    
    def cancel_order(self, order_id: int) -> bool:
        """取消订单"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        book = self.bids if order.side == OrderSide.BUY else self.asks
        
        if order.price in book:
            book[order.price] = [o for o in book[order.price] if o.order_id != order_id]
            if not book[order.price]:
                del book[order.price]
        
        del self.orders[order_id]
        return True
    
    def get_best_bid(self) -> Optional[float]:
        """最优买价"""
        return max(self.bids.keys()) if self.bids else None
    
    def get_best_ask(self) -> Optional[float]:
        """最优卖价"""
        return min(self.asks.keys()) if self.asks else None
    
    def get_mid_price(self) -> Optional[float]:
        """中间价"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None
    
    def get_spread(self) -> Optional[float]:
        """买卖价差"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None
    
    def get_depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        """获取订单簿深度"""
        bid_prices = sorted(self.bids.keys(), reverse=True)[:levels]
        ask_prices = sorted(self.asks.keys())[:levels]
        
        bids = [(p, sum(o.quantity for o in self.bids[p])) for p in bid_prices]
        asks = [(p, sum(o.quantity for o in self.asks[p])) for p in ask_prices]
        
        return {'bids': bids, 'asks': asks}
    
    def match_orders(self) -> List[Trade]:
        """撮合订单"""
        trades = []
        
        while self.bids and self.asks:
            best_bid_price = self.get_best_bid()
            best_ask_price = self.get_best_ask()
            
            if best_bid_price is None or best_ask_price is None:
                break
            
            if best_bid_price < best_ask_price:
                break
            
            # 取出最优买卖单
            bid_orders = self.bids[best_bid_price]
            ask_orders = self.asks[best_ask_price]
            
            bid_order = bid_orders[0]
            ask_order = ask_orders[0]
            
            # 成交价格（取时间优先的价格）
            trade_price = bid_order.price if bid_order.timestamp < ask_order.timestamp else ask_order.price
            
            # 成交数量
            trade_quantity = min(bid_order.quantity, ask_order.quantity)
            
            # 记录成交
            trade = Trade(
                buyer_id=bid_order.agent_id,
                seller_id=ask_order.agent_id,
                price=trade_price,
                quantity=trade_quantity,
                timestamp=max(bid_order.timestamp, ask_order.timestamp)
            )
            trades.append(trade)
            
            # 更新订单数量
            bid_order.quantity -= trade_quantity
            ask_order.quantity -= trade_quantity
            
            # 移除完全成交的订单
            if bid_order.quantity <= 0:
                bid_orders.pop(0)
                del self.orders[bid_order.order_id]
                if not bid_orders:
                    del self.bids[best_bid_price]
            
            if ask_order.quantity <= 0:
                ask_orders.pop(0)
                del self.orders[ask_order.order_id]
                if not ask_orders:
                    del self.asks[best_ask_price]
        
        return trades


class MarketEnvironment:
    """
    市场环境模拟器
    支持多智能体交互、订单簿机制、市场冲击、制度约束
    """
    def __init__(
        self,
        num_agents: int = 5,
        initial_price: float = 100.0,
        tick_size: float = 0.01,
        max_position: float = 1000.0,
        transaction_cost: float = 0.001,  # 0.1%
        price_impact_coef: float = 0.01,
        volatility: float = 0.02,
        # 制度约束
        price_limit_pct: float = 0.10,  # 涨跌停 10%
        circuit_breaker_pct: float = 0.05,  # 熔断 5%
        max_order_size: float = 100.0,
        min_order_size: float = 0.1,
    ):
        self.num_agents = num_agents
        self.initial_price = initial_price
        self.current_price = initial_price
        self.tick_size = tick_size
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.price_impact_coef = price_impact_coef
        self.volatility = volatility
        
        # 制度约束
        self.price_limit_pct = price_limit_pct
        self.circuit_breaker_pct = circuit_breaker_pct
        self.max_order_size = max_order_size
        self.min_order_size = min_order_size
        
        # 订单簿
        self.order_book = OrderBook(tick_size)
        
        # 智能体状态
        self.positions = np.zeros(num_agents)  # 持仓
        self.cash = np.ones(num_agents) * 10000.0  # 现金
        self.pnl = np.zeros(num_agents)  # 盈亏
        
        # 市场历史
        self.price_history = [initial_price]
        self.volume_history = [0.0]
        self.trade_history: List[Trade] = []
        
        # 时间
        self.current_time = 0
        
        # 熔断状态
        self.is_halted = False
        self.halt_time = 0
        
        # 参考价格（用于���跌停计算）
        self.reference_price = initial_price
        
    def reset(self) -> Dict[str, np.ndarray]:
        """重置环境"""
        self.current_price = self.initial_price
        self.order_book = OrderBook(self.tick_size)
        self.positions = np.zeros(self.num_agents)
        self.cash = np.ones(self.num_agents) * 10000.0
        self.pnl = np.zeros(self.num_agents)
        self.price_history = [self.initial_price]
        self.volume_history = [0.0]
        self.trade_history = []
        self.current_time = 0
        self.is_halted = False
        self.halt_time = 0
        self.reference_price = self.initial_price
        
        return self.get_state()
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """获取当前市场状态"""
        # 订单簿特征
        depth = self.order_book.get_depth(levels=5)
        bid_prices = np.array([p for p, _ in depth['bids']] + [0] * (5 - len(depth['bids'])))
        bid_volumes = np.array([v for _, v in depth['bids']] + [0] * (5 - len(depth['bids'])))
        ask_prices = np.array([p for p, _ in depth['asks']] + [0] * (5 - len(depth['asks'])))
        ask_volumes = np.array([v for _, v in depth['asks']] + [0] * (5 - len(depth['asks'])))
        
        # 价格特征
        mid_price = self.order_book.get_mid_price() or self.current_price
        spread = self.order_book.get_spread() or 0.0
        
        # 历史价格特征
        recent_prices = np.array(self.price_history[-20:] + [self.current_price] * (20 - len(self.price_history)))
        returns = np.diff(recent_prices) / (recent_prices[:-1] + 1e-8)
        
        # 成交量特征
        recent_volumes = np.array(self.volume_history[-10:] + [0] * (10 - len(self.volume_history)))
        
        state = {
            'price': np.array([self.current_price]),
            'mid_price': np.array([mid_price]),
            'spread': np.array([spread]),
            'bid_prices': bid_prices,
            'bid_volumes': bid_volumes,
            'ask_prices': ask_prices,
            'ask_volumes': ask_volumes,
            'returns': returns,
            'volumes': recent_volumes,
            'positions': self.positions.copy(),
            'cash': self.cash.copy(),
            'pnl': self.pnl.copy(),
            'time': np.array([self.current_time]),
            'is_halted': np.array([float(self.is_halted)]),
            'price_limit_upper': np.array([self.reference_price * (1 + self.price_limit_pct)]),
            'price_limit_lower': np.array([self.reference_price * (1 - self.price_limit_pct)]),
        }
        
        return state
    
    def get_state_vector(self, agent_id: int) -> np.ndarray:
        """获取状态向量（用于神经网络输入）"""
        state = self.get_state()
        
        # 拼接所有特征
        features = np.concatenate([
            state['price'],
            state['mid_price'],
            state['spread'],
            state['bid_prices'],
            state['bid_volumes'],
            state['ask_prices'],
            state['ask_volumes'],
            state['returns'],
            state['volumes'],
            state['positions'][[agent_id]],  # 自己的持仓
            state['cash'][[agent_id]],  # 自己的现金
            state['pnl'][[agent_id]],  # 自己的盈亏
            state['time'],
            state['is_halted'],
            state['price_limit_upper'],
            state['price_limit_lower'],
        ])
        
        return features
    
    def check_constraints(self, agent_id: int, order: Order) -> Tuple[bool, str]:
        """检查订单是否满足约束"""
        # 检查熔断
        if self.is_halted:
            return False, "Market is halted"
        
        # 检查订单大小
        if order.quantity < self.min_order_size:
            return False, f"Order size too small (min: {self.min_order_size})"
        if order.quantity > self.max_order_size:
            return False, f"Order size too large (max: {self.max_order_size})"
        
        # 检查涨跌停
        upper_limit = self.reference_price * (1 + self.price_limit_pct)
        lower_limit = self.reference_price * (1 - self.price_limit_pct)
        
        if order.price > upper_limit:
            return False, f"Price exceeds upper limit ({upper_limit:.2f})"
        if order.price < lower_limit:
            return False, f"Price below lower limit ({lower_limit:.2f})"
        
        # 检查持仓限制
        if order.side == OrderSide.BUY:
            new_position = self.positions[agent_id] + order.quantity
            if new_position > self.max_position:
                return False, f"Position limit exceeded (max: {self.max_position})"
            
            # 检查现金
            required_cash = order.price * order.quantity * (1 + self.transaction_cost)
            if self.cash[agent_id] < required_cash:
                return False, "Insufficient cash"
        else:
            new_position = self.positions[agent_id] - order.quantity
            if new_position < -self.max_position:
                return False, f"Position limit exceeded (max: {self.max_position})"
            
            # 检查持仓（不能卖空超过限制）
            if self.positions[agent_id] < order.quantity:
                return False, "Insufficient position"
        
        return True, "OK"
    
    def compute_market_impact(self, quantity: float, side: OrderSide) -> float:
        """计算市场冲击"""
        # 简单的线性冲击模型
        impact = self.price_impact_coef * quantity
        return impact if side == OrderSide.BUY else -impact
    
    def step(self, actions: List[Dict]) -> Tuple[Dict[str, np.ndarray], np.ndarray, bool, Dict]:
        """
        执行一步
        
        Args:
            actions: 每个智能体的动作列表
                每个动作是字典: {'type': 'limit/market', 'side': 'buy/sell', 'price': float, 'quantity': float}
        
        Returns:
            next_state: 下一个状态
            rewards: 每个智能体的奖励
            done: 是否结束
            info: 额外信息
        """
        self.current_time += 1
        
        # 检查熔断恢复
        if self.is_halted and self.current_time - self.halt_time > 10:
            self.is_halted = False
        
        # 处理每个智能体的动作
        valid_orders = []
        rejected_orders = []
        
        for agent_id, action in enumerate(actions):
            if action is None or self.is_halted:
                continue
            
            # 解析动作
            order_type = OrderType.LIMIT if action.get('type') == 'limit' else OrderType.MARKET
            side = OrderSide.BUY if action.get('side') == 'buy' else OrderSide.SELL
            price = action.get('price', self.current_price)
            quantity = action.get('quantity', 0.0)
            
            if quantity <= 0:
                continue
            
            order = Order(
                agent_id=agent_id,
                order_type=order_type,
                side=side,
                price=price,
                quantity=quantity,
                timestamp=self.current_time
            )
            
            # 检查约束
            is_valid, message = self.check_constraints(agent_id, order)
            
            if is_valid:
                self.order_book.add_order(order)
                valid_orders.append(order)
            else:
                rejected_orders.append((agent_id, message))
        
        # 撮合订单
        trades = self.order_book.match_orders()
        self.trade_history.extend(trades)
        
        # 更新持仓和现金
        for trade in trades:
            # 买方
            self.positions[trade.buyer_id] += trade.quantity
            cost = trade.price * trade.quantity * (1 + self.transaction_cost)
            self.cash[trade.buyer_id] -= cost
            
            # 卖方
            self.positions[trade.seller_id] -= trade.quantity
            revenue = trade.price * trade.quantity * (1 - self.transaction_cost)
            self.cash[trade.seller_id] += revenue
        
        # 更新价格
        if trades:
            total_volume = sum(t.quantity for t in trades)
            weighted_price = sum(t.price * t.quantity for t in trades) / total_volume
            self.current_price = weighted_price
            self.volume_history.append(total_volume)
        else:
            # 添加随机波动
            self.current_price *= (1 + np.random.normal(0, self.volatility))
            self.volume_history.append(0.0)
        
        self.price_history.append(self.current_price)
        
        # 检查熔断
        price_change_pct = abs(self.current_price - self.reference_price) / self.reference_price
        if price_change_pct >= self.circuit_breaker_pct and not self.is_halted:
            self.is_halted = True
            self.halt_time = self.current_time
        
        # 更新参考价格（每天）
        if self.current_time % 100 == 0:
            self.reference_price = self.current_price
        
        # 计算盈亏
        for agent_id in range(self.num_agents):
            position_value = self.positions[agent_id] * self.current_price
            self.pnl[agent_id] = self.cash[agent_id] + position_value - 10000.0
        
        # 计算奖励（基于盈亏变化）
        rewards = self.pnl.copy()
        
        # 获取下一个状态
        next_state = self.get_state()
        
        # 结束条件
        done = self.current_time >= 1000 or any(self.cash < 0)
        
        # 额外信息
        info = {
            'trades': trades,
            'valid_orders': len(valid_orders),
            'rejected_orders': rejected_orders,
            'is_halted': self.is_halted,
            'price': self.current_price,
            'volume': self.volume_history[-1],
        }
        
        return next_state, rewards, done, info
    
    def render(self):
        """可视化市场状态"""
        print(f"\n=== Time: {self.current_time} ===")
        print(f"Price: {self.current_price:.2f}")
        print(f"Mid Price: {self.order_book.get_mid_price():.2f if self.order_book.get_mid_price() else 'N/A'}")
        print(f"Spread: {self.order_book.get_spread():.4f if self.order_book.get_spread() else 'N/A'}")
        print(f"Halted: {self.is_halted}")
        
        depth = self.order_book.get_depth(levels=3)
        print("\nOrder Book:")
        print("  Asks:")
        for price, volume in reversed(depth['asks']):
            print(f"    {price:.2f} | {volume:.2f}")
        print("  ---")
        print("  Bids:")
        for price, volume in depth['bids']:
            print(f"    {price:.2f} | {volume:.2f}")
        
        print("\nAgent States:")
        for i in range(self.num_agents):
            print(f"  Agent {i}: Position={self.positions[i]:.2f}, Cash={self.cash[i]:.2f}, PnL={self.pnl[i]:.2f}")


if __name__ == '__main__':
    # 测试市场环境
    env = MarketEnvironment(num_agents=3)
    state = env.reset()
    
    print("Initial state keys:", state.keys())
    print("State vector shape:", env.get_state_vector(0).shape)
    
    # 模拟几步
    for step in range(5):
        # 随机动作
        actions = []
        for agent_id in range(3):
            if np.random.random() > 0.5:
                action = {
                    'type': 'limit',
                    'side': 'buy' if np.random.random() > 0.5 else 'sell',
                    'price': env.current_price * (1 + np.random.uniform(-0.01, 0.01)),
                    'quantity': np.random.uniform(1, 10)
                }
            else:
                action = None
            actions.append(action)
        
        next_state, rewards, done, info = env.step(actions)
        env.render()
        
        if done:
            break
    
    print("\n✓ Market environment test passed!")

