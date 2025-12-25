"""
Market Simulation with Diffusion Agent
使用训练好的 Diffusion 模型进行市场模拟
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import argparse
from pathlib import Path

from models.diffusion.market_diffusion import MarketDiffusionModel
from models.market.environment import MarketEnvironment
from models.agents.game_agents import AgentFactory, DiffusionAgent


class MarketSimulator:
    """市场模拟器"""
    def __init__(
        self,
        env: MarketEnvironment,
        agents: List,
        save_dir: str = 'results'
    ):
        self.env = env
        self.agents = agents
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 记录
        self.price_history = []
        self.volume_history = []
        self.wealth_history = {i: [] for i in range(len(agents))}
        self.action_history = {i: [] for i in range(len(agents))}
        
    def run_episode(self, num_steps: int = 500, render: bool = False) -> Dict:
        """运行一个回合"""
        state = self.env.reset()
        
        for step in range(num_steps):
            # 收集动作
            actions = []
            for agent in self.agents:
                action = agent.act(state, {})
                actions.append(action)
                
                # 记录动作
                if action is not None:
                    self.action_history[agent.agent_id].append(action)
            
            # 执行动作
            next_state, rewards, done, info = self.env.step(actions)
            
            # 记录
            self.price_history.append(info['price'])
            self.volume_history.append(info['volume'])
            
            for i, agent in enumerate(self.agents):
                wealth = state['cash'][i] + state['positions'][i] * info['price']
                self.wealth_history[i].append(wealth)
            
            # 渲染
            if render and step % 50 == 0:
                self.env.render()
            
            state = next_state
            
            if done:
                break
        
        # 计算统计信息
        stats = self._compute_statistics()
        
        return stats
    
    def _compute_statistics(self) -> Dict:
        """计算统计信息"""
        stats = {
            'price_mean': np.mean(self.price_history),
            'price_std': np.std(self.price_history),
            'price_min': np.min(self.price_history),
            'price_max': np.max(self.price_history),
            'total_volume': np.sum(self.volume_history),
            'avg_volume': np.mean(self.volume_history),
        }
        
        # 每个智能体的统计
        for i, agent in enumerate(self.agents):
            initial_wealth = 10000.0
            final_wealth = self.wealth_history[i][-1] if self.wealth_history[i] else initial_wealth
            pnl = final_wealth - initial_wealth
            returns = pnl / initial_wealth
            
            stats[f'agent_{i}_type'] = agent.agent_type
            stats[f'agent_{i}_pnl'] = pnl
            stats[f'agent_{i}_returns'] = returns
            stats[f'agent_{i}_num_actions'] = len(self.action_history[i])
        
        return stats
    
    def plot_results(self):
        """绘制结果"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 价格轨迹
        axes[0, 0].plot(self.price_history, linewidth=1)
        axes[0, 0].set_title('Price Trajectory')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 成交量
        axes[0, 1].bar(range(len(self.volume_history)), self.volume_history, alpha=0.6)
        axes[0, 1].set_title('Trading Volume')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Volume')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 财富轨迹
        for i, agent in enumerate(self.agents):
            if self.wealth_history[i]:
                axes[1, 0].plot(self.wealth_history[i], label=f'Agent {i} ({agent.agent_type})', linewidth=1)
        axes[1, 0].set_title('Wealth Trajectory')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Wealth')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 收益率分布
        returns = []
        labels = []
        for i, agent in enumerate(self.agents):
            if self.wealth_history[i]:
                initial = 10000.0
                final = self.wealth_history[i][-1]
                ret = (final - initial) / initial * 100
                returns.append(ret)
                labels.append(f'Agent {i}\n({agent.agent_type})')
        
        axes[1, 1].bar(range(len(returns)), returns, alpha=0.6)
        axes[1, 1].set_title('Returns by Agent')
        axes[1, 1].set_xticks(range(len(returns)))
        axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Returns (%)')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 价格收益率分布
        price_returns = np.diff(self.price_history) / np.array(self.price_history[:-1])
        axes[2, 0].hist(price_returns, bins=50, alpha=0.6, edgecolor='black')
        axes[2, 0].set_title('Price Returns Distribution')
        axes[2, 0].set_xlabel('Returns')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        axes[2, 0].grid(True, alpha=0.3)
        
        # 动作数量
        action_counts = [len(self.action_history[i]) for i in range(len(self.agents))]
        axes[2, 1].bar(range(len(action_counts)), action_counts, alpha=0.6)
        axes[2, 1].set_title('Number of Actions by Agent')
        axes[2, 1].set_xticks(range(len(action_counts)))
        axes[2, 1].set_xticklabels([f'Agent {i}' for i in range(len(action_counts))])
        axes[2, 1].set_ylabel('Number of Actions')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'simulation_results.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {self.save_dir / 'simulation_results.png'}")
        plt.close()
    
    def print_summary(self, stats: Dict):
        """打印摘要"""
        print("\n" + "="*60)
        print("SIMULATION SUMMARY")
        print("="*60)
        
        print("\nMarket Statistics:")
        print(f"  Price Mean: {stats['price_mean']:.2f}")
        print(f"  Price Std: {stats['price_std']:.2f}")
        print(f"  Price Range: [{stats['price_min']:.2f}, {stats['price_max']:.2f}]")
        print(f"  Total Volume: {stats['total_volume']:.2f}")
        print(f"  Avg Volume: {stats['avg_volume']:.2f}")
        
        print("\nAgent Performance:")
        for i in range(len(self.agents)):
            agent_type = stats[f'agent_{i}_type']
            pnl = stats[f'agent_{i}_pnl']
            returns = stats[f'agent_{i}_returns'] * 100
            num_actions = stats[f'agent_{i}_num_actions']
            
            print(f"  Agent {i} ({agent_type}):")
            print(f"    PnL: ${pnl:.2f}")
            print(f"    Returns: {returns:.2f}%")
            print(f"    Actions: {num_actions}")
        
        print("="*60)


def load_model(checkpoint_path: str, device: str = 'cuda') -> MarketDiffusionModel:
    """加载训练好的模型"""
    print(f"Loading model from {checkpoint_path}...")
    
    # 创建模型
    model = MarketDiffusionModel(
        action_dim=10,
        state_dim=50,  # 需要根据实际情况调整
        opponent_dim=40,
        constraint_dim=20,
        num_opponents=4,
        hidden_dim=512,
        num_diffusion_steps=1000,
        beta_schedule='cosine'
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Simulate market with Diffusion agent")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--num_agents', type=int, default=5, help='Number of agents')
    parser.add_argument('--num_steps', type=int, default=500, help='Number of simulation steps')
    parser.add_argument('--num_episodes', type=int, default=1, help='Number of episodes')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='results', help='Save directory')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建环境
    print("\nCreating market environment...")
    env = MarketEnvironment(
        num_agents=args.num_agents,
        initial_price=100.0,
        volatility=0.02
    )
    
    # 创建智能体
    print("Creating agents...")
    if args.checkpoint and Path(args.checkpoint).exists():
        # 加载 Diffusion 模型
        diffusion_model = load_model(args.checkpoint, device)
        agents = AgentFactory.create_mixed_population(args.num_agents, diffusion_model)
        print(f"✓ Created {args.num_agents} agents (1 Diffusion + {args.num_agents-1} traditional)")
    else:
        # 只使用传统智能体
        agents = AgentFactory.create_mixed_population(args.num_agents, None)
        print(f"✓ Created {args.num_agents} traditional agents")
    
    # 打印智能体类型
    for i, agent in enumerate(agents):
        print(f"  Agent {i}: {agent.agent_type}")
    
    # 运行模拟
    print(f"\nRunning {args.num_episodes} episode(s)...")
    
    for episode in range(args.num_episodes):
        print(f"\n--- Episode {episode + 1}/{args.num_episodes} ---")
        
        simulator = MarketSimulator(env, agents, args.save_dir)
        stats = simulator.run_episode(num_steps=args.num_steps, render=args.render)
        
        # 打印摘要
        simulator.print_summary(stats)
        
        # 绘制结果
        simulator.plot_results()
    
    print("\n✓ Simulation completed!")


if __name__ == '__main__':
    main()

