"""
Quick Start Example
快速开始示例：展示如何使用 DiffFinance
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion.market_diffusion import MarketDiffusionModel
from models.market.environment import MarketEnvironment
from models.agents.game_agents import AgentFactory
from models.game_theory.nash_solver import NashEquilibriumSolver


def example_1_basic_simulation():
    """示例 1: 基础市场模���"""
    print("\n" + "="*60)
    print("Example 1: Basic Market Simulation")
    print("="*60)
    
    # 创建市场环境
    env = MarketEnvironment(
        num_agents=4,
        initial_price=100.0,
        volatility=0.02
    )
    
    # 创建不同类型的智能体
    agents = [
        AgentFactory.create_agent('market_maker', 0),
        AgentFactory.create_agent('speculator', 1),
        AgentFactory.create_agent('arbitrageur', 2),
        AgentFactory.create_agent('informed_trader', 3),
    ]
    
    # 运行模拟
    state = env.reset()
    
    for step in range(50):
        # 收集动作
        actions = [agent.act(state, {}) for agent in agents]
        
        # 执行
        next_state, rewards, done, info = env.step(actions)
        
        if step % 10 == 0:
            print(f"\nStep {step}:")
            print(f"  Price: ${info['price']:.2f}")
            print(f"  Volume: {info['volume']:.2f}")
            print(f"  Halted: {info['is_halted']}")
        
        state = next_state
        
        if done:
            break
    
    print("\n✓ Basic simulation completed!")


def example_2_diffusion_model():
    """示例 2: Diffusion 模型训练和推理"""
    print("\n" + "="*60)
    print("Example 2: Diffusion Model")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建模型
    model = MarketDiffusionModel(
        action_dim=10,
        state_dim=50,
        opponent_dim=40,
        constraint_dim=20,
        num_opponents=4,
        hidden_dim=256,  # 较小的模型用于演示
        num_diffusion_steps=100  # 较少的步数用于演示
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 模拟数据
    batch_size = 4
    action = torch.randn(batch_size, 10).to(device)
    market_state = torch.randn(batch_size, 50).to(device)
    opponent_strategy = torch.randn(batch_size, 40).to(device)
    constraints = torch.randn(batch_size, 20).to(device)
    
    # 前向传播（训练）
    print("\nTraining forward pass...")
    outputs = model(action, market_state, opponent_strategy, constraints)
    losses = model.compute_loss(outputs)
    
    print(f"  Diffusion loss: {losses['diffusion_loss'].item():.4f}")
    print(f"  Impact magnitude: {losses['impact_magnitude'].item():.4f}")
    
    # 采样（推理）
    print("\nSampling strategies...")
    with torch.no_grad():
        generated_actions = model.sample(
            market_state[:1],
            opponent_strategy[:1],
            constraints[:1],
            num_samples=3
        )
    
    print(f"  Generated {generated_actions.shape[0]} strategies")
    print(f"  Action shape: {generated_actions.shape}")
    
    print("\n✓ Diffusion model example completed!")


def example_3_nash_equilibrium():
    """示例 3: 纳什均衡求解"""
    print("\n" + "="*60)
    print("Example 3: Nash Equilibrium")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = MarketDiffusionModel(
        action_dim=10,
        state_dim=50,
        opponent_dim=40,
        constraint_dim=20,
        num_opponents=3,
        hidden_dim=128,
        num_diffusion_steps=50
    ).to(device)
    
    # 创建求解器
    solver = NashEquilibriumSolver(model, num_agents=4, device=device)
    
    # 模拟市场状态
    market_state = torch.randn(1, 50).to(device)
    constraints = torch.randn(1, 20).to(device)
    
    # 求解纳什均衡
    print("\nSolving Nash equilibrium...")
    equilibrium_strategies, converged = solver.find_nash_equilibrium(
        market_state,
        constraints,
        max_iterations=5,  # 较少迭代用于演示
        tolerance=1e-2
    )
    
    print(f"  Converged: {converged}")
    print(f"  Number of strategies: {len(equilibrium_strategies)}")
    
    # 计算均衡收益
    payoffs = solver.compute_nash_equilibrium_payoffs(equilibrium_strategies, market_state)
    print(f"  Equilibrium payoffs: {payoffs}")
    
    print("\n✓ Nash equilibrium example completed!")


def example_4_self_impact_analysis():
    """示例 4: 自我影响力分析"""
    print("\n" + "="*60)
    print("Example 4: Self-Impact Analysis")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = MarketDiffusionModel(
        action_dim=10,
        state_dim=50,
        opponent_dim=40,
        constraint_dim=20
    ).to(device)
    
    # 模拟不同规模的订单
    market_state = torch.randn(1, 50).to(device)
    
    print("\nAnalyzing market impact for different order sizes:")
    
    for quantity in [1.0, 5.0, 10.0, 20.0]:
        # 构造动作（买单）
        action = torch.zeros(1, 10).to(device)
        action[0, 1] = 1.0  # 买入
        action[0, 3] = quantity / 10.0  # 数量（归一化）
        
        # 预测市场影响
        with torch.no_grad():
            predicted_impact, impact_magnitude = model.self_impact(action, market_state)
        
        print(f"  Order size: {quantity:.1f}")
        print(f"    Impact magnitude: {impact_magnitude.item():.4f}")
        print(f"    Predicted price change: {predicted_impact[0, 0].item():.4f}")
    
    print("\n✓ Self-impact analysis completed!")


def example_5_opponent_response():
    """示例 5: 对手响应预测"""
    print("\n" + "="*60)
    print("Example 5: Opponent Response Prediction")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = MarketDiffusionModel(
        action_dim=10,
        state_dim=50,
        opponent_dim=40,
        constraint_dim=20,
        num_opponents=3
    ).to(device)
    
    # 模拟自己的动作
    my_action = torch.randn(1, 10).to(device)
    market_state = torch.randn(1, 50).to(device)
    
    # 预测对手响应
    print("\nPredicting opponent responses...")
    with torch.no_grad():
        responses, opponent_types = model.opponent_response(my_action, market_state)
    
    print(f"  Number of opponents: {responses.shape[1]}")
    print(f"  Response shape: {responses.shape}")
    print(f"  Opponent type probabilities:")
    for i, prob in enumerate(opponent_types[0]):
        print(f"    Opponent {i}: {prob.item():.3f}")
    
    print("\n✓ Opponent response prediction completed!")


def example_6_complete_workflow():
    """示例 6: 完整工作流程"""
    print("\n" + "="*60)
    print("Example 6: Complete Workflow")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 创建环境和智能体
    print("\n1. Creating environment and agents...")
    env = MarketEnvironment(num_agents=3)
    agents = AgentFactory.create_mixed_population(3, None)
    
    # 2. 收集数据
    print("\n2. Collecting training data...")
    samples = []
    state = env.reset()
    
    for step in range(20):
        for agent in agents:
            action = agent.act(state, {})
            if action is not None:
                samples.append({
                    'action': action,
                    'state': state,
                    'agent_id': agent.agent_id
                })
        
        actions = [agent.act(state, {}) for agent in agents]
        next_state, rewards, done, info = env.step(actions)
        state = next_state
        
        if done:
            break
    
    print(f"  Collected {len(samples)} samples")
    
    # 3. 创建和训练模型（演示）
    print("\n3. Creating model...")
    model = MarketDiffusionModel(
        action_dim=10,
        state_dim=50,
        opponent_dim=40,
        constraint_dim=20,
        num_opponents=2,
        hidden_dim=128,
        num_diffusion_steps=50
    ).to(device)
    
    # 4. 使用模型生成策略
    print("\n4. Generating strategies with model...")
    market_state = torch.randn(1, 50).to(device)
    opponent_strategy = torch.randn(1, 40).to(device)
    constraints = torch.randn(1, 20).to(device)
    
    with torch.no_grad():
        generated_strategy = model.sample(
            market_state,
            opponent_strategy,
            constraints,
            num_samples=1
        )
    
    print(f"  Generated strategy shape: {generated_strategy.shape}")
    
    # 5. 评估策略
    print("\n5. Evaluating strategy...")
    print(f"  Strategy vector: {generated_strategy[0, :5].cpu().numpy()}")
    
    print("\n✓ Complete workflow example completed!")


def main():
    """运行所有示例"""
    print("\n" + "="*70)
    print(" "*15 + "DiffFinance Quick Start Examples")
    print("="*70)
    
    examples = [
        ("Basic Market Simulation", example_1_basic_simulation),
        ("Diffusion Model", example_2_diffusion_model),
        ("Nash Equilibrium", example_3_nash_equilibrium),
        ("Self-Impact Analysis", example_4_self_impact_analysis),
        ("Opponent Response", example_5_opponent_response),
        ("Complete Workflow", example_6_complete_workflow),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...\n")
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    
    print("\nNext steps:")
    print("  1. Train a model: python training/train_market_diffusion.py")
    print("  2. Run simulation: python inference/simulate_market.py")
    print("  3. Evaluate strategies: python evaluation/evaluate_strategies.py")
    print("\nFor more information, see README.md")


if __name__ == '__main__':
    main()
