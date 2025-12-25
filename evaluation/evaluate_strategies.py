"""
Evaluation and Backtesting
评估策略性能和回测
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import argparse
from pathlib import Path

from models.diffusion.market_diffusion import MarketDiffusionModel
from models.market.environment import MarketEnvironment
from models.agents.game_agents import AgentFactory, DiffusionAgent


class StrategyEvaluator:
    """策略评估器"""
    def __init__(self, save_dir: str = 'evaluation'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_agent(
        self,
        agent,
        env: MarketEnvironment,
        num_episodes: int = 100,
        episode_length: int = 500
    ) -> Dict:
        """评估单个智能体"""
        all_returns = []
        all_sharpe_ratios = []
        all_max_drawdowns = []
        all_win_rates = []
        
        for episode in range(num_episodes):
            state = env.reset()
            wealth_history = [10000.0]
            
            for step in range(episode_length):
                # 智能体动作
                action = agent.act(state, {})
                actions = [action if i == agent.agent_id else None for i in range(env.num_agents)]
                
                # 执行
                next_state, rewards, done, info = env.step(actions)
                
                # 记录财富
                wealth = state['cash'][agent.agent_id] + state['positions'][agent.agent_id] * info['price']
                wealth_history.append(wealth)
                
                state = next_state
                
                if done:
                    break
            
            # 计算指标
            wealth_array = np.array(wealth_history)
            returns = np.diff(wealth_array) / wealth_array[:-1]
            
            # 总收益率
            total_return = (wealth_array[-1] - wealth_array[0]) / wealth_array[0]
            all_returns.append(total_return)
            
            # Sharpe 比率
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe = 0.0
            all_sharpe_ratios.append(sharpe)
            
            # 最大回撤
            cummax = np.maximum.accumulate(wealth_array)
            drawdown = (wealth_array - cummax) / cummax
            max_drawdown = np.min(drawdown)
            all_max_drawdowns.append(max_drawdown)
            
            # 胜率
            win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0.0
            all_win_rates.append(win_rate)
        
        # 汇总统计
        metrics = {
            'mean_return': np.mean(all_returns),
            'std_return': np.std(all_returns),
            'mean_sharpe': np.mean(all_sharpe_ratios),
            'mean_max_drawdown': np.mean(all_max_drawdowns),
            'mean_win_rate': np.mean(all_win_rates),
            'returns': all_returns,
            'sharpe_ratios': all_sharpe_ratios,
            'max_drawdowns': all_max_drawdowns,
            'win_rates': all_win_rates,
        }
        
        return metrics
    
    def compare_agents(
        self,
        agents: List,
        env: MarketEnvironment,
        num_episodes: int = 50
    ) -> pd.DataFrame:
        """比较多个智能体"""
        results = []
        
        for agent in agents:
            print(f"\nEvaluating {agent.agent_type} (Agent {agent.agent_id})...")
            metrics = self.evaluate_agent(agent, env, num_episodes)
            
            results.append({
                'agent_id': agent.agent_id,
                'agent_type': agent.agent_type,
                'mean_return': metrics['mean_return'],
                'std_return': metrics['std_return'],
                'sharpe_ratio': metrics['mean_sharpe'],
                'max_drawdown': metrics['mean_max_drawdown'],
                'win_rate': metrics['mean_win_rate'],
            })
        
        df = pd.DataFrame(results)
        
        # 保存
        df.to_csv(self.save_dir / 'agent_comparison.csv', index=False)
        print(f"\n✓ Saved comparison to {self.save_dir / 'agent_comparison.csv'}")
        
        return df
    
    def plot_comparison(self, df: pd.DataFrame):
        """绘制比较图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        metrics = ['mean_return', 'std_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        titles = ['Mean Return', 'Std Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            row = idx // 3
            col = idx % 3
            
            ax = axes[row, col]
            
            # 按智能体类型分组
            df_sorted = df.sort_values(metric, ascending=False)
            
            colors = ['#1f77b4' if t != 'diffusion_agent' else '#ff7f0e' 
                     for t in df_sorted['agent_type']]
            
            ax.barh(range(len(df_sorted)), df_sorted[metric], color=colors, alpha=0.7)
            ax.set_yticks(range(len(df_sorted)))
            ax.set_yticklabels([f"{row['agent_type']}\n(Agent {row['agent_id']})" 
                                for _, row in df_sorted.iterrows()], fontsize=8)
            ax.set_xlabel(title)
            ax.grid(True, alpha=0.3, axis='x')
            
            # 添加数值标签
            for i, v in enumerate(df_sorted[metric]):
                ax.text(v, i, f' {v:.3f}', va='center', fontsize=8)
        
        # 删除多余的子图
        fig.delaxes(axes[1, 2])
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', alpha=0.7, label='Traditional Agent'),
            Patch(facecolor='#ff7f0e', alpha=0.7, label='Diffusion Agent')
        ]
        axes[1, 2].legend(handles=legend_elements, loc='center', fontsize=12)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'agent_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {self.save_dir / 'agent_comparison.png'}")
        plt.close()
    
    def analyze_market_impact(
        self,
        agent,
        env: MarketEnvironment,
        num_episodes: int = 20
    ) -> Dict:
        """分析智能体对市场的影响"""
        price_impacts = []
        volume_impacts = []
        volatility_impacts = []
        
        for episode in range(num_episodes):
            state = env.reset()
            initial_price = env.current_price
            initial_volatility = env.volatility
            
            episode_volumes = []
            
            for step in range(100):
                action = agent.act(state, {})
                actions = [action if i == agent.agent_id else None for i in range(env.num_agents)]
                
                next_state, rewards, done, info = env.step(actions)
                
                if action is not None:
                    episode_volumes.append(action.get('quantity', 0))
                
                state = next_state
                
                if done:
                    break
            
            # 价格影响
            final_price = env.current_price
            price_impact = (final_price - initial_price) / initial_price
            price_impacts.append(price_impact)
            
            # 成交量影响
            avg_volume = np.mean(episode_volumes) if episode_volumes else 0
            volume_impacts.append(avg_volume)
            
            # 波动率影响
            price_returns = np.diff(env.price_history) / np.array(env.price_history[:-1])
            realized_vol = np.std(price_returns) if len(price_returns) > 0 else 0
            volatility_impacts.append(realized_vol)
        
        return {
            'mean_price_impact': np.mean(price_impacts),
            'mean_volume_impact': np.mean(volume_impacts),
            'mean_volatility_impact': np.mean(volatility_impacts),
            'price_impacts': price_impacts,
            'volume_impacts': volume_impacts,
            'volatility_impacts': volatility_impacts,
        }


class BacktestEngine:
    """回测引擎"""
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        
    def backtest(
        self,
        agent,
        price_data: np.ndarray,
        features: np.ndarray
    ) -> Dict:
        """
        回测策略
        
        Args:
            agent: 智能体
            price_data: 价格序列
            features: 特征矩阵
        
        Returns:
            backtest_results: 回测结果
        """
        cash = self.initial_capital
        position = 0.0
        wealth_history = [self.initial_capital]
        position_history = [0.0]
        trade_history = []
        
        for t in range(len(price_data) - 1):
            current_price = price_data[t]
            
            # 构造状态（简化）
            state = {
                'price': np.array([current_price]),
                'mid_price': np.array([current_price]),
                'spread': np.array([0.01]),
                'bid_prices': np.zeros(5),
                'bid_volumes': np.zeros(5),
                'ask_prices': np.zeros(5),
                'ask_volumes': np.zeros(5),
                'returns': features[max(0, t-19):t+1, 0] if t >= 19 else np.zeros(20),
                'volumes': np.zeros(10),
                'positions': np.array([position]),
                'cash': np.array([cash]),
                'pnl': np.array([cash + position * current_price - self.initial_capital]),
                'time': np.array([t]),
                'is_halted': np.array([0.0]),
                'price_limit_upper': np.array([current_price * 1.1]),
                'price_limit_lower': np.array([current_price * 0.9]),
            }
            
            # 智能体决策
            action = agent.act(state, {})
            
            if action is not None:
                side = action['side']
                quantity = action['quantity']
                
                # 执行交易
                if side == 'buy' and cash >= quantity * current_price:
                    position += quantity
                    cash -= quantity * current_price * 1.001  # 交易成本
                    trade_history.append({
                        'time': t,
                        'side': 'buy',
                        'price': current_price,
                        'quantity': quantity
                    })
                elif side == 'sell' and position >= quantity:
                    position -= quantity
                    cash += quantity * current_price * 0.999  # 交易成本
                    trade_history.append({
                        'time': t,
                        'side': 'sell',
                        'price': current_price,
                        'quantity': quantity
                    })
            
            # 记录
            wealth = cash + position * price_data[t + 1]
            wealth_history.append(wealth)
            position_history.append(position)
        
        # 计算指标
        wealth_array = np.array(wealth_history)
        returns = np.diff(wealth_array) / wealth_array[:-1]
        
        total_return = (wealth_array[-1] - wealth_array[0]) / wealth_array[0]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        cummax = np.maximum.accumulate(wealth_array)
        drawdown = (wealth_array - cummax) / cummax
        max_drawdown = np.min(drawdown)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(trade_history),
            'wealth_history': wealth_history,
            'position_history': position_history,
            'trade_history': trade_history,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate strategies")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--num_agents', type=int, default=5, help='Number of agents')
    parser.add_argument('--num_episodes', type=int, default=50, help='Number of evaluation episodes')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--save_dir', type=str, default='evaluation', help='Save directory')
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建环境
    print("\nCreating environment...")
    env = MarketEnvironment(num_agents=args.num_agents)
    
    # 创建智能体
    print("Creating agents...")
    if args.checkpoint and Path(args.checkpoint).exists():
        # 加载 Diffusion 模型
        from inference.simulate_market import load_model
        diffusion_model = load_model(args.checkpoint, device)
        agents = AgentFactory.create_mixed_population(args.num_agents, diffusion_model)
    else:
        agents = AgentFactory.create_mixed_population(args.num_agents, None)
    
    # 创建评估器
    evaluator = StrategyEvaluator(save_dir=args.save_dir)
    
    # 比较智能体
    print("\nComparing agents...")
    df = evaluator.compare_agents(agents, env, num_episodes=args.num_episodes)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    # 绘制比较图
    evaluator.plot_comparison(df)
    
    # 分析市场影响（仅对 Diffusion 智能体）
    diffusion_agents = [a for a in agents if a.agent_type == 'diffusion_agent']
    if diffusion_agents:
        print("\nAnalyzing market impact of Diffusion agent...")
        impact = evaluator.analyze_market_impact(diffusion_agents[0], env, num_episodes=20)
        
        print(f"  Mean price impact: {impact['mean_price_impact']:.4f}")
        print(f"  Mean volume impact: {impact['mean_volume_impact']:.2f}")
        print(f"  Mean volatility impact: {impact['mean_volatility_impact']:.4f}")
    
    print("\n✓ Evaluation completed!")


if __name__ == '__main__':
    main()

