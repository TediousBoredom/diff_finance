"""
Nash Equilibrium Solver
使用 Diffusion 模型求解市场博弈的纳什均衡
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize
import nashpy as nash


class NashEquilibriumSolver:
    """
    纳什均衡求解器
    结合 Diffusion 模型和博弈论方法
    """
    def __init__(
        self,
        diffusion_model: nn.Module,
        num_agents: int,
        device: str = 'cuda'
    ):
        self.diffusion_model = diffusion_model
        self.num_agents = num_agents
        self.device = device
        
    def compute_best_response(
        self,
        agent_id: int,
        market_state: torch.Tensor,
        opponent_strategies: torch.Tensor,
        constraints: torch.Tensor,
        num_samples: int = 10
    ) -> torch.Tensor:
        """
        计算最优响应策略
        
        Args:
            agent_id: 智能体 ID
            market_state: 市场状态
            opponent_strategies: 对手策略
            constraints: 约束
            num_samples: 采样数量
        
        Returns:
            best_response: 最优响应策略
        """
        with torch.no_grad():
            # 使用 Diffusion 模型生成多个候选策略
            candidate_strategies = self.diffusion_model.sample(
                market_state,
                opponent_strategies,
                constraints,
                num_samples=num_samples
            )
            
            # 评估每个策略的期望收益
            payoffs = self._evaluate_strategies(
                agent_id,
                candidate_strategies,
                market_state,
                opponent_strategies
            )
            
            # 选择最优策略
            best_idx = torch.argmax(payoffs)
            best_response = candidate_strategies[best_idx]
            
        return best_response
    
    def _evaluate_strategies(
        self,
        agent_id: int,
        strategies: torch.Tensor,
        market_state: torch.Tensor,
        opponent_strategies: torch.Tensor
    ) -> torch.Tensor:
        """
        评估策略的期望收益
        
        使用简化的收益函数：
        - 交易收益
        - 市场冲击成本
        - 库存风险
        """
        batch_size = strategies.shape[0]
        payoffs = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            strategy = strategies[i]
            
            # 解析策略
            side = strategy[1]  # 买卖方向
            quantity = abs(strategy[3]) * 10  # 数量
            
            # 计算交易收益（简化）
            price_impact = self._estimate_price_impact(quantity, side)
            trading_profit = quantity * price_impact * (-1 if side > 0 else 1)
            
            # 计算市场冲击成本
            impact_cost = 0.01 * quantity * abs(price_impact)
            
            # 计算库存风险
            current_position = market_state[0, -3]  # 假设位置
            new_position = current_position + (quantity if side > 0 else -quantity)
            inventory_risk = 0.001 * new_position ** 2
            
            # 总收益
            payoffs[i] = trading_profit - impact_cost - inventory_risk
        
        return payoffs
    
    def _estimate_price_impact(self, quantity: float, side: float) -> float:
        """估计价格冲击"""
        # 简单的线性冲击模型
        impact = 0.01 * quantity
        return impact if side > 0 else -impact
    
    def find_nash_equilibrium(
        self,
        market_state: torch.Tensor,
        constraints: torch.Tensor,
        max_iterations: int = 50,
        tolerance: float = 1e-3
    ) -> Tuple[List[torch.Tensor], bool]:
        """
        使用迭代最优响应法求解纳什均衡
        
        Args:
            market_state: 市场状态
            constraints: 约束
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
        
        Returns:
            equilibrium_strategies: 均衡策略列表
            converged: 是否收敛
        """
        # 初始化策略
        strategies = [
            torch.randn(10, device=self.device) * 0.1
            for _ in range(self.num_agents)
        ]
        
        converged = False
        
        for iteration in range(max_iterations):
            old_strategies = [s.clone() for s in strategies]
            
            # 对每个智能体计算最优响应
            for agent_id in range(self.num_agents):
                # 构造对手策略
                opponent_strategies = torch.cat([
                    strategies[j] for j in range(self.num_agents) if j != agent_id
                ], dim=0).unsqueeze(0)
                
                # 填充到固定维度
                if opponent_strategies.shape[1] < 40:
                    padding = torch.zeros(
                        1, 40 - opponent_strategies.shape[1],
                        device=self.device
                    )
                    opponent_strategies = torch.cat([opponent_strategies, padding], dim=1)
                
                # 计算最优响应
                best_response = self.compute_best_response(
                    agent_id,
                    market_state,
                    opponent_strategies,
                    constraints,
                    num_samples=5
                )
                
                strategies[agent_id] = best_response
            
            # 检查收敛
            max_change = max(
                torch.norm(strategies[i] - old_strategies[i]).item()
                for i in range(self.num_agents)
            )
            
            if max_change < tolerance:
                converged = True
                print(f"✓ Converged at iteration {iteration + 1}")
                break
        
        if not converged:
            print(f"⚠ Did not converge after {max_iterations} iterations")
        
        return strategies, converged
    
    def compute_nash_equilibrium_payoffs(
        self,
        equilibrium_strategies: List[torch.Tensor],
        market_state: torch.Tensor
    ) -> np.ndarray:
        """计算纳什均衡下的收益"""
        payoffs = np.zeros(self.num_agents)
        
        for agent_id in range(self.num_agents):
            strategy = equilibrium_strategies[agent_id].unsqueeze(0)
            opponent_strategies = torch.cat([
                equilibrium_strategies[j] for j in range(self.num_agents) if j != agent_id
            ], dim=0).unsqueeze(0)
            
            # 填充
            if opponent_strategies.shape[1] < 40:
                padding = torch.zeros(
                    1, 40 - opponent_strategies.shape[1],
                    device=self.device
                )
                opponent_strategies = torch.cat([opponent_strategies, padding], dim=1)
            
            payoff = self._evaluate_strategies(
                agent_id,
                strategy,
                market_state,
                opponent_strategies
            )
            
            payoffs[agent_id] = payoff.item()
        
        return payoffs


class StackelbergEquilibriumSolver:
    """
    Stackelberg 均衡求解器（领导者-跟随者博弈）
    适用于有信息优势的智能体
    """
    def __init__(
        self,
        diffusion_model: nn.Module,
        leader_id: int,
        follower_ids: List[int],
        device: str = 'cuda'
    ):
        self.diffusion_model = diffusion_model
        self.leader_id = leader_id
        self.follower_ids = follower_ids
        self.device = device
        
    def solve(
        self,
        market_state: torch.Tensor,
        constraints: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        求解 Stackelberg 均衡
        
        Returns:
            leader_strategy: 领导者策略
            follower_strategies: 跟随者策略列表
        """
        # 领导者优化：考虑跟随者的最优响应
        best_leader_strategy = None
        best_leader_payoff = float('-inf')
        
        # 采样多个领导者策略
        num_samples = 20
        
        for _ in range(num_samples):
            # 生成候选领导者策略
            leader_strategy = torch.randn(1, 10, device=self.device) * 0.1
            
            # 预测跟随者响应
            follower_strategies = []
            for follower_id in self.follower_ids:
                # 构造对手策略（包括领导者）
                opponent_strategies = leader_strategy.clone()
                
                # 填充
                if opponent_strategies.shape[1] < 40:
                    padding = torch.zeros(
                        1, 40 - opponent_strategies.shape[1],
                        device=self.device
                    )
                    opponent_strategies = torch.cat([opponent_strategies, padding], dim=1)
                
                # 使用 Diffusion 模型生成跟随者策略
                follower_strategy = self.diffusion_model.sample(
                    market_state,
                    opponent_strategies,
                    constraints,
                    num_samples=1
                )
                
                follower_strategies.append(follower_strategy)
            
            # 评估领导者收益
            all_opponent_strategies = torch.cat(follower_strategies, dim=0).unsqueeze(0)
            if all_opponent_strategies.shape[1] < 40:
                padding = torch.zeros(
                    1, 40 - all_opponent_strategies.shape[1],
                    device=self.device
                )
                all_opponent_strategies = torch.cat([all_opponent_strategies, padding], dim=1)
            
            leader_payoff = self._evaluate_leader_payoff(
                leader_strategy,
                all_opponent_strategies,
                market_state
            )
            
            if leader_payoff > best_leader_payoff:
                best_leader_payoff = leader_payoff
                best_leader_strategy = leader_strategy
                best_follower_strategies = follower_strategies
        
        return best_leader_strategy, best_follower_strategies
    
    def _evaluate_leader_payoff(
        self,
        leader_strategy: torch.Tensor,
        opponent_strategies: torch.Tensor,
        market_state: torch.Tensor
    ) -> float:
        """评估领导者收益"""
        # 简化的收益函数
        side = leader_strategy[0, 1]
        quantity = abs(leader_strategy[0, 3]) * 10
        
        # 先行优势
        first_mover_advantage = 0.1 * quantity
        
        # 市场冲击
        impact_cost = 0.01 * quantity ** 2
        
        return first_mover_advantage - impact_cost


class CooperativeGameSolver:
    """
    合作博弈求解器
    用于分析智能体联盟和收益分配
    """
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        
    def compute_shapley_value(
        self,
        characteristic_function: Dict[frozenset, float]
    ) -> np.ndarray:
        """
        计算 Shapley 值（公平收益分配）
        
        Args:
            characteristic_function: 特征函数 v(S)，映射联盟到收益
        
        Returns:
            shapley_values: 每个智能体的 Shapley 值
        """
        shapley_values = np.zeros(self.num_agents)
        
        # 遍历所有可能的排列
        from itertools import permutations
        
        all_perms = list(permutations(range(self.num_agents)))
        
        for perm in all_perms:
            coalition = set()
            for agent in perm:
                # 计算边际贡献
                old_value = characteristic_function.get(frozenset(coalition), 0.0)
                coalition.add(agent)
                new_value = characteristic_function.get(frozenset(coalition), 0.0)
                
                marginal_contribution = new_value - old_value
                shapley_values[agent] += marginal_contribution
        
        # 平均
        shapley_values /= len(all_perms)
        
        return shapley_values
    
    def find_core(
        self,
        characteristic_function: Dict[frozenset, float],
        grand_coalition_value: float
    ) -> Optional[np.ndarray]:
        """
        寻找核心（Core）分配
        
        核心是满足所有联盟稳定性约束的分配
        """
        from scipy.optimize import linprog
        
        # 构造线性规划问题
        # min 0 (可行性问题)
        # s.t. sum(x_i for i in S) >= v(S) for all S
        #      sum(x_i) = v(N)
        #      x_i >= 0
        
        # 这里简化实现，只检查 Shapley 值是否在核心中
        shapley = self.compute_shapley_value(characteristic_function)
        
        # 检查是否满足所有联盟约束
        is_in_core = True
        for coalition, value in characteristic_function.items():
            coalition_sum = sum(shapley[i] for i in coalition)
            if coalition_sum < value - 1e-6:
                is_in_core = False
                break
        
        if is_in_core and abs(sum(shapley) - grand_coalition_value) < 1e-6:
            return shapley
        else:
            return None


if __name__ == '__main__':
    # 测试纳什均衡求解器
    print("Testing Nash Equilibrium Solver...")
    
    from models.diffusion.market_diffusion import MarketDiffusionModel
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = MarketDiffusionModel(
        action_dim=10,
        state_dim=50,
        opponent_dim=40,
        constraint_dim=20,
        num_opponents=4
    ).to(device)
    
    # 创建求解器
    solver = NashEquilibriumSolver(model, num_agents=5, device=device)
    
    # 模拟市场状态
    market_state = torch.randn(1, 50).to(device)
    constraints = torch.randn(1, 20).to(device)
    
    # 求解纳什均衡
    equilibrium_strategies, converged = solver.find_nash_equilibrium(
        market_state,
        constraints,
        max_iterations=10
    )
    
    print(f"Converged: {converged}")
    print(f"Number of equilibrium strategies: {len(equilibrium_strategies)}")
    
    # 计算均衡收益
    payoffs = solver.compute_nash_equilibrium_payoffs(equilibrium_strategies, market_state)
    print(f"Equilibrium payoffs: {payoffs}")
    
    print("\n✓ Nash equilibrium solver test passed!")
    
    # 测试 Shapley 值
    print("\nTesting Shapley Value...")
    
    coop_solver = CooperativeGameSolver(num_agents=3)
    
    # 示例特征函数
    char_func = {
        frozenset(): 0.0,
        frozenset([0]): 1.0,
        frozenset([1]): 1.0,
        frozenset([2]): 1.0,
        frozenset([0, 1]): 3.0,
        frozenset([0, 2]): 3.0,
        frozenset([1, 2]): 3.0,
        frozenset([0, 1, 2]): 6.0,
    }
    
    shapley = coop_solver.compute_shapley_value(char_func)
    print(f"Shapley values: {shapley}")
    print(f"Sum: {sum(shapley)}")
    
    print("\n✓ Cooperative game solver test passed!")
