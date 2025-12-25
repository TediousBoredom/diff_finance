"""
Training script for Market Diffusion Model
训练基于 Diffusion 的市场策略生成模型
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from typing import Dict, List, Tuple
import argparse
from pathlib import Path

from models.diffusion.market_diffusion import MarketDiffusionModel
from models.market.environment import MarketEnvironment
from models.agents.game_agents import AgentFactory


class MarketTrajectoryDataset(Dataset):
    """市场轨迹数据集"""
    def __init__(
        self,
        num_episodes: int = 1000,
        episode_length: int = 100,
        num_agents: int = 5
    ):
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.num_agents = num_agents
        
        print(f"Generating {num_episodes} market episodes...")
        self.data = self._generate_data()
        print(f"Generated {len(self.data)} samples")
    
    def _generate_data(self) -> List[Dict]:
        """生成训练数据"""
        env = MarketEnvironment(num_agents=self.num_agents)
        agents = AgentFactory.create_mixed_population(self.num_agents)
        
        all_samples = []
        
        for episode in tqdm(range(self.num_episodes), desc="Generating episodes"):
            state = env.reset()
            
            for step in range(self.episode_length):
                # 收集每个智能体的动作
                for agent in agents:
                    action = agent.act(state, {})
                    
                    if action is not None:
                        # 构造样本
                        sample = {
                            'agent_id': agent.agent_id,
                            'market_state': env.get_state_vector(agent.agent_id),
                            'action': self._action_to_vector(action, state['price'][0]),
                            'opponent_strategy': self._get_opponent_features(state, agent.agent_id),
                            'constraints': self._get_constraint_features(state, agent.agent_id),
                            'reward': state['pnl'][agent.agent_id],
                        }
                        all_samples.append(sample)
                
                # 执行动作
                actions = [agent.act(state, {}) for agent in agents]
                next_state, rewards, done, info = env.step(actions)
                state = next_state
                
                if done:
                    break
        
        return all_samples
    
    def _action_to_vector(self, action: Dict, current_price: float) -> np.ndarray:
        """将动作转换为向量"""
        # 动作向量: [order_type, side, price_offset, quantity, ...]
        order_type = 1.0 if action['type'] == 'limit' else -1.0
        side = 1.0 if action['side'] == 'buy' else -1.0
        price_offset = (action['price'] - current_price) / current_price
        quantity = action['quantity'] / 10.0  # 归一化
        
        # 填充到固定维度
        vector = np.array([order_type, side, price_offset, quantity] + [0.0] * 6)
        return vector
    
    def _get_opponent_features(self, state: Dict, agent_id: int) -> np.ndarray:
        """获取对手特征"""
        opponent_positions = np.delete(state['positions'], agent_id)
        opponent_pnl = np.delete(state['pnl'], agent_id)
        
        features = np.concatenate([opponent_positions, opponent_pnl])
        # 填充到固定维度
        features = np.pad(features, (0, max(0, 40 - len(features))), mode='constant')[:40]
        return features
    
    def _get_constraint_features(self, state: Dict, agent_id: int) -> np.ndarray:
        """获取约束特征"""
        features = np.concatenate([
            state['positions'][[agent_id]],
            state['cash'][[agent_id]] / 10000.0,  # 归一化
            state['is_halted'],
            [state['price_limit_upper'][0] / state['price'][0]],
            [state['price_limit_lower'][0] / state['price'][0]],
        ])
        # 填充到固定维度
        features = np.pad(features, (0, max(0, 20 - len(features))), mode='constant')[:20]
        return features
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        return {
            'action': torch.FloatTensor(sample['action']),
            'market_state': torch.FloatTensor(sample['market_state']),
            'opponent_strategy': torch.FloatTensor(sample['opponent_strategy']),
            'constraints': torch.FloatTensor(sample['constraints']),
            'reward': torch.FloatTensor([sample['reward']]),
        }


class Trainer:
    """训练器"""
    def __init__(
        self,
        model: MarketDiffusionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        lr: float = 1e-4,
        use_wandb: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_wandb = use_wandb
        
        # 优化器
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 100,
            eta_min=1e-6
        )
        
        # 最佳模型
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_diffusion_loss = 0.0
        total_impact = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 移动到设备
            action = batch['action'].to(self.device)
            market_state = batch['market_state'].to(self.device)
            opponent_strategy = batch['opponent_strategy'].to(self.device)
            constraints = batch['constraints'].to(self.device)
            
            # 前向传播
            outputs = self.model(action, market_state, opponent_strategy, constraints)
            losses = self.model.compute_loss(outputs)
            
            # 反向传播
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 记录
            total_loss += losses['total_loss'].item()
            total_diffusion_loss += losses['diffusion_loss'].item()
            total_impact += losses['impact_magnitude'].item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': losses['total_loss'].item(),
                'diff_loss': losses['diffusion_loss'].item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # 记录到 wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': losses['total_loss'].item(),
                    'train/diffusion_loss': losses['diffusion_loss'].item(),
                    'train/impact_magnitude': losses['impact_magnitude'].item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                })
        
        num_batches = len(self.train_loader)
        return {
            'loss': total_loss / num_batches,
            'diffusion_loss': total_diffusion_loss / num_batches,
            'impact_magnitude': total_impact / num_batches,
        }
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        total_diffusion_loss = 0.0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            # 移动到设备
            action = batch['action'].to(self.device)
            market_state = batch['market_state'].to(self.device)
            opponent_strategy = batch['opponent_strategy'].to(self.device)
            constraints = batch['constraints'].to(self.device)
            
            # 前向传播
            outputs = self.model(action, market_state, opponent_strategy, constraints)
            losses = self.model.compute_loss(outputs)
            
            total_loss += losses['total_loss'].item()
            total_diffusion_loss += losses['diffusion_loss'].item()
        
        num_batches = len(self.val_loader)
        val_metrics = {
            'loss': total_loss / num_batches,
            'diffusion_loss': total_diffusion_loss / num_batches,
        }
        
        # 记录到 wandb
        if self.use_wandb:
            wandb.log({
                'val/loss': val_metrics['loss'],
                'val/diffusion_loss': val_metrics['diffusion_loss'],
                'epoch': epoch,
            })
        
        return val_metrics
    
    def save_checkpoint(self, epoch: int, save_dir: str, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存最新检查点
        torch.save(checkpoint, save_path / f'checkpoint_epoch_{epoch}.pt')
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, save_path / 'best_model.pt')
            print(f"✓ Saved best model at epoch {epoch}")
    
    def train(self, num_epochs: int, save_dir: str):
        """完整训练流程"""
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            
            # 验证
            val_metrics = self.validate(epoch)
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            
            # 保存检查点
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, save_dir, is_best)
        
        print("\n✓ Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Market Diffusion Model")
    parser.add_argument('--num_episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--episode_length', type=int, default=100, help='Length of each episode')
    parser.add_argument('--num_agents', type=int, default=5, help='Number of agents')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='difffinance', help='W&B project name')
    
    args = parser.parse_args()
    
    # 初始化 wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"market_diffusion_{args.num_episodes}ep"
        )
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建数据集
    print("\nCreating datasets...")
    train_dataset = MarketTrajectoryDataset(
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        num_agents=args.num_agents
    )
    
    val_dataset = MarketTrajectoryDataset(
        num_episodes=args.num_episodes // 5,
        episode_length=args.episode_length,
        num_agents=args.num_agents
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    print("\nCreating model...")
    # 获取实际的状态维度
    sample = train_dataset[0]
    state_dim = sample['market_state'].shape[0]
    
    model = MarketDiffusionModel(
        action_dim=10,
        state_dim=state_dim,
        opponent_dim=40,
        constraint_dim=20,
        num_opponents=args.num_agents - 1,
        hidden_dim=512,
        num_diffusion_steps=1000,
        beta_schedule='cosine'
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        use_wandb=args.use_wandb
    )
    
    # 训练
    trainer.train(num_epochs=args.num_epochs, save_dir=args.save_dir)
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

