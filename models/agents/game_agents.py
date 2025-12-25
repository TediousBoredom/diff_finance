"""
Game Theory Agents: 不同类型的市场参与者
包括：做市商、投机者、套利者、知情交易者
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """基础智能体类"""
    def __init__(self, agent_id: int, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.wealth_history = []
        self.action_history = []
        
    @abstractmethod
    def act(self, state: Dict[str, np.ndarray], market_info: Dict) -> Dict:
        """根据状态选择动作"""
        pass
    
    def update(self, reward: float, next_state: Dict[str, np.ndarray]):
        """更新智能体（用于学习）"""
        pass
    
    def reset(self):
        """重置智能体状态"""
        self.wealth_history = []
        self.action_history = []


class MarketMaker(BaseAgent):
    """
    做市商：提供流动性，赚取买卖价差
    策略：在买卖两侧同时挂单，控制库存风险
    """
    def __init__(
        self,
        agent_id: int,
        target_spread: float = 0.02,
        max_inventory: float = 100.0,
        risk_aversion: float = 0.5
    ):
        super().__init__(agent_id, "market_maker")
        self.target_spread = target_spread
        self.max_inventory = max_inventory
        self.risk_aversion = risk_aversion
        
    def act(self, state: Dict[str, np.ndarray], market_info: Dict) -> Dict:
        """
        做市商策略：
        1. 根据库存调整报价（库存过多时降低卖价、提高买价）
        2. 根据市场波动调整价差
        3. 控制订单大小
        """
        current_price = state['price'][0]
        position = state['positions'][self.agent_id]
        mid_price = state['mid_price'][0]
        
        # 计算库存偏离
        inventory_skew = position / self.max_inventory
        
        # 调整价差（库存越多，价差越大）
        spread = self.target_spread * (1 + abs(inventory_skew) * self.risk_aversion)
        
        # 计算报价
        bid_price = mid_price - spread / 2 - inventory_skew * spread
        ask_price = mid_price + spread / 2 - inventory_skew * spread
        
        # 订单大小（库存越多，卖单越大）
        bid_quantity = max(1.0, self.max_inventory * 0.1 * (1 - inventory_skew))
        ask_quantity = max(1.0, self.max_inventory * 0.1 * (1 + inventory_skew))
        
        # 随机选择买或卖（或两者都做）
        action_type = np.random.choice(['buy', 'sell', 'both'])
        
        if action_type == 'buy' or action_type == 'both':
            return {
                'type': 'limit',
                'side': 'buy',
                'price': bid_price,
                'quantity': bid_quantity
            }
        else:
            return {
                'type': 'limit',
                'side': 'sell',
                'price': ask_price,
                'quantity': ask_quantity
            }


class Speculator(BaseAgent):
    """
    投机者：基于价格趋势和动量进行交易
    策略：趋势跟随 + 均值回归
    """
    def __init__(
        self,
        agent_id: int,
        momentum_window: int = 10,
        mean_reversion_threshold: float = 0.02,
        position_limit: float = 50.0
    ):
        super().__init__(agent_id, "speculator")
        self.momentum_window = momentum_window
        self.mean_reversion_threshold = mean_reversion_threshold
        self.position_limit = position_limit
        
    def act(self, state: Dict[str, np.ndarray], market_info: Dict) -> Dict:
        """
        投机者策略：
        1. 计算价格动量
        2. 检测均值回归机会
        3. 根据信号强度调整仓位
        """
        current_price = state['price'][0]
        returns = state['returns']
        position = state['positions'][self.agent_id]
        
        # 计算动量
        if len(returns) >= self.momentum_window:
            momentum = np.mean(returns[-self.momentum_window:])
        else:
            momentum = 0.0
        
        # 计算价格偏离（简单均值回归）
        if len(returns) > 0:
            mean_price = np.mean(state['price'])
            deviation = (current_price - mean_price) / mean_price
        else:
            deviation = 0.0
        
        # 生成信号
        signal = 0.0
        
        # 动量信号
        if momentum > 0.001:
            signal += 1.0
        elif momentum < -0.001:
            signal -= 1.0
        
        # 均值回归信号
        if abs(deviation) > self.mean_reversion_threshold:
            signal -= np.sign(deviation) * 0.5
        
        # 根据信号决定动作
        if signal > 0.3 and position < self.position_limit:
            # 买入
            quantity = min(10.0, self.position_limit - position)
            return {
                'type': 'market',
                'side': 'buy',
                'price': current_price * 1.001,  # 略高于市价
                'quantity': quantity
            }
        elif signal < -0.3 and position > -self.position_limit:
            # 卖出
            quantity = min(10.0, self.position_limit + position)
            return {
                'type': 'market',
                'side': 'sell',
                'price': current_price * 0.999,  # 略低于市价
                'quantity': quantity
            }
        else:
            # 不交易
            return None


class Arbitrageur(BaseAgent):
    """
    套利者：寻找价格偏差进行套利
    策略：统计套利、价差交易
    """
    def __init__(
        self,
        agent_id: int,
        fair_value_window: int = 20,
        arbitrage_threshold: float = 0.01,
        max_position: float = 30.0
    ):
        super().__init__(agent_id, "arbitrageur")
        self.fair_value_window = fair_value_window
        self.arbitrage_threshold = arbitrage_threshold
        self.max_position = max_position
        
    def act(self, state: Dict[str, np.ndarray], market_info: Dict) -> Dict:
        """
        套利者策略：
        1. 估计公允价值
        2. 检测价格偏离
        3. 快速平仓获利
        """
        current_price = state['price'][0]
        mid_price = state['mid_price'][0]
        position = state['positions'][self.agent_id]
        
        # 估计公允价值（使用历史均价）
        price_history = state.get('price_history', [current_price])
        if len(price_history) >= self.fair_value_window:
            fair_value = np.mean(price_history[-self.fair_value_window:])
        else:
            fair_value = current_price
        
        # 计算价格偏离
        mispricing = (current_price - fair_value) / fair_value
        
        # 套利信号
        if mispricing < -self.arbitrage_threshold and position < self.max_position:
            # 价格低于公允价值，买入
            quantity = min(5.0, self.max_position - position)
            return {
                'type': 'limit',
                'side': 'buy',
                'price': current_price * 1.0005,
                'quantity': quantity
            }
        elif mispricing > self.arbitrage_threshold and position > -self.max_position:
            # 价格高于公允价值，卖出
            quantity = min(5.0, self.max_position + position)
            return {
                'type': 'limit',
                'side': 'sell',
                'price': current_price * 0.9995,
                'quantity': quantity
            }
        elif abs(position) > 0 and abs(mispricing) < self.arbitrage_threshold * 0.5:
            # 平仓
            if position > 0:
                return {
                    'type': 'market',
                    'side': 'sell',
                    'price': current_price * 0.999,
                    'quantity': min(abs(position), 5.0)
                }
            else:
                return {
                    'type': 'market',
                    'side': 'buy',
                    'price': current_price * 1.001,
                    'quantity': min(abs(position), 5.0)
                }
        
        return None


class InformedTrader(BaseAgent):
    """
    知情交易者：拥有私有信息，战略性交易
    策略：隐藏信息、分批交易、避免市场冲击
    """
    def __init__(
        self,
        agent_id: int,
        private_signal_strength: float = 0.05,
        stealth_factor: float = 0.3,
        max_position: float = 100.0
    ):
        super().__init__(agent_id, "informed_trader")
        self.private_signal_strength = private_signal_strength
        self.stealth_factor = stealth_factor
        self.max_position = max_position
        self.private_signal = 0.0
        self.signal_duration = 0
        
    def generate_private_signal(self):
        """生成私有信息信号"""
        # 随机生成一个持续一段时间的信号
        if self.signal_duration <= 0:
            self.private_signal = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            self.signal_duration = np.random.randint(10, 30)
        else:
            self.signal_duration -= 1
    
    def act(self, state: Dict[str, np.ndarray], market_info: Dict) -> Dict:
        """
        知情交易者策略：
        1. 利用私有信息
        2. 分批交易以隐藏意图
        3. 避免过大的市场冲击
        """
        self.generate_private_signal()
        
        current_price = state['price'][0]
        position = state['positions'][self.agent_id]
        spread = state['spread'][0]
        
        # 根据私有信号决定目标仓位
        target_position = self.private_signal * self.max_position
        
        # 计算需要调整的仓位
        position_diff = target_position - position
        
        if abs(position_diff) < 1.0:
            return None
        
        # 分批交易（隐蔽性）
        trade_size = abs(position_diff) * self.stealth_factor
        trade_size = max(1.0, min(trade_size, 10.0))
        
        # 使用限价单以减少市场冲击
        if position_diff > 0:
            # 需要买入
            # 在买一价附近挂单
            price = current_price - spread * 0.3
            return {
                'type': 'limit',
                'side': 'buy',
                'price': price,
                'quantity': trade_size
            }
        else:
            # 需要卖出
            # 在卖一价附近挂单
            price = current_price + spread * 0.3
            return {
                'type': 'limit',
                'side': 'sell',
                'price': price,
                'quantity': trade_size
            }


class DiffusionAgent(BaseAgent):
    """
    基于 Diffusion 模型的智能体
    使用训练好的 Diffusion 模型生成最优策略
    """
    def __init__(
        self,
        agent_id: int,
        diffusion_model: nn.Module,
        device: str = 'cpu'
    ):
        super().__init__(agent_id, "diffusion_agent")
        self.diffusion_model = diffusion_model
        self.device = device
        self.diffusion_model.eval()
        
    def state_to_tensor(self, state: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """将状态转换为模型输入"""
        # 市场状态
        market_features = np.concatenate([
            state['price'],
            state['mid_price'],
            state['spread'],
            state['bid_prices'],
            state['bid_volumes'],
            state['ask_prices'],
            state['ask_volumes'],
            state['returns'],
            state['volumes'][:5],  # 取前5个
        ])
        
        # 对手策略（其他智能体的持仓）
        opponent_positions = np.delete(state['positions'], self.agent_id)
        opponent_features = np.concatenate([
            opponent_positions,
            np.zeros(40 - len(opponent_positions))  # 填充到固定维度
        ])
        
        # 约束
        constraint_features = np.concatenate([
            state['positions'][[self.agent_id]],
            state['cash'][[self.agent_id]],
            state['is_halted'],
            state['price_limit_upper'],
            state['price_limit_lower'],
            np.zeros(15)  # 填充
        ])
        
        # 转换为 tensor
        market_state = torch.FloatTensor(market_features).unsqueeze(0).to(self.device)
        opponent_strategy = torch.FloatTensor(opponent_features).unsqueeze(0).to(self.device)
        constraints = torch.FloatTensor(constraint_features).unsqueeze(0).to(self.device)
        
        return market_state, opponent_strategy, constraints
    
    def tensor_to_action(self, action_tensor: torch.Tensor, current_price: float) -> Dict:
        """将模型输出转换为动作"""
        action_array = action_tensor.cpu().numpy()[0]
        
        # 解析动作向量
        # action_array: [order_type, side, price_offset, quantity, ...]
        order_type_logit = action_array[0]
        side_logit = action_array[1]
        price_offset = action_array[2]
        quantity = abs(action_array[3])
        
        # 决定订单类型
        order_type = 'limit' if order_type_logit > 0 else 'market'
        
        # 决定买卖方向
        side = 'buy' if side_logit > 0 else 'sell'
        
        # 计算价格
        price = current_price * (1 + np.clip(price_offset, -0.01, 0.01))
        
        # 限制数量
        quantity = np.clip(quantity * 10, 0.1, 20.0)
        
        return {
            'type': order_type,
            'side': side,
            'price': price,
            'quantity': quantity
        }
    
    @torch.no_grad()
    def act(self, state: Dict[str, np.ndarray], market_info: Dict) -> Dict:
        """使用 Diffusion 模型生成动作"""
        # 转换状态
        market_state, opponent_strategy, constraints = self.state_to_tensor(state)
        
        # 生成动作
        action_tensor = self.diffusion_model.sample(
            market_state,
            opponent_strategy,
            constraints,
            num_samples=1
        )
        
        # 转换为动作字典
        current_price = state['price'][0]
        action = self.tensor_to_action(action_tensor, current_price)
        
        return action


class AgentFactory:
    """智能体工厂"""
    @staticmethod
    def create_agent(agent_type: str, agent_id: int, **kwargs) -> BaseAgent:
        """创建智能体"""
        if agent_type == "market_maker":
            return MarketMaker(agent_id, **kwargs)
        elif agent_type == "speculator":
            return Speculator(agent_id, **kwargs)
        elif agent_type == "arbitrageur":
            return Arbitrageur(agent_id, **kwargs)
        elif agent_type == "informed_trader":
            return InformedTrader(agent_id, **kwargs)
        elif agent_type == "diffusion_agent":
            return DiffusionAgent(agent_id, **kwargs)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    @staticmethod
    def create_mixed_population(num_agents: int, diffusion_model: Optional[nn.Module] = None) -> List[BaseAgent]:
        """创建混合智能体群体"""
        agents = []
        
        # 分配智能体类型
        types = ['market_maker', 'speculator', 'arbitrageur', 'informed_trader']
        
        for i in range(num_agents):
            if diffusion_model is not None and i == 0:
                # 第一个智能体使用 Diffusion 模型
                agent = DiffusionAgent(i, diffusion_model)
            else:
                # 其他智能体随机选择类型
                agent_type = np.random.choice(types)
                agent = AgentFactory.create_agent(agent_type, i)
            
            agents.append(agent)
        
        return agents


if __name__ == '__main__':
    # 测试智能体
    from models.market.environment import MarketEnvironment
    
    env = MarketEnvironment(num_agents=4)
    state = env.reset()
    
    # 创建不同类型的智能体
    agents = [
        MarketMaker(0),
        Speculator(1),
        Arbitrageur(2),
        InformedTrader(3)
    ]
    
    print("Testing agents...")
    for step in range(10):
        actions = []
        for agent in agents:
            action = agent.act(state, {})
            actions.append(action)
            if action:
                print(f"Agent {agent.agent_id} ({agent.agent_type}): {action}")
        
        next_state, rewards, done, info = env.step(actions)
        state = next_state
        
        if done:
            break
    
    print("\n✓ Agents test passed!")

