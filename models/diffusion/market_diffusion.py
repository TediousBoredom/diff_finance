"""
Market Strategy Diffusion Model
基于 Diffusion 的市场策略生成模型，支持条件生成和博弈论约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """时间步嵌入"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class MarketConditionEncoder(nn.Module):
    """市场条件编码器：编码市场状态、对手策略、制度约束"""
    def __init__(
        self,
        state_dim: int,
        opponent_dim: int,
        constraint_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 512
    ):
        super().__init__()
        
        # 市场状态编码
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 对手策略编码
        self.opponent_encoder = nn.Sequential(
            nn.Linear(opponent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 制度约束编码
        self.constraint_encoder = nn.Sequential(
            nn.Linear(constraint_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(
        self,
        market_state: torch.Tensor,
        opponent_strategy: torch.Tensor,
        constraints: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            market_state: [batch, state_dim] 市场状态（价格、成交量、订单簿等）
            opponent_strategy: [batch, opponent_dim] 对手策略
            constraints: [batch, constraint_dim] 制度约束
        Returns:
            condition_embedding: [batch, output_dim]
        """
        state_emb = self.state_encoder(market_state)
        opponent_emb = self.opponent_encoder(opponent_strategy)
        constraint_emb = self.constraint_encoder(constraints)
        
        # 拼接并融合
        combined = torch.cat([state_emb, opponent_emb, constraint_emb], dim=-1)
        condition_embedding = self.fusion(combined)
        
        return condition_embedding


class SelfImpactModule(nn.Module):
    """自我影响力建模模块：量化自身决策对市场的影响"""
    def __init__(self, action_dim: int, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.impact_predictor = nn.Sequential(
            nn.Linear(action_dim + state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),  # 预测状态变化
        )
        
        # 影响力权重（可学习）
        self.impact_weight = nn.Parameter(torch.ones(1))
        
    def forward(
        self,
        action: torch.Tensor,
        current_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            action: [batch, action_dim] 当前动作
            current_state: [batch, state_dim] 当前市场状态
        Returns:
            predicted_impact: [batch, state_dim] 预测的市场影响
            impact_magnitude: [batch, 1] 影响力大小
        """
        x = torch.cat([action, current_state], dim=-1)
        predicted_impact = self.impact_predictor(x)
        
        # 计算影响力大小
        impact_magnitude = torch.norm(predicted_impact, dim=-1, keepdim=True)
        impact_magnitude = impact_magnitude * self.impact_weight
        
        return predicted_impact, impact_magnitude


class OpponentResponseModule(nn.Module):
    """对手响应预测模块：预测其他参与者对自身行动的反应"""
    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        num_opponents: int = 4,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.num_opponents = num_opponents
        
        # 对每个对手建模
        self.response_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(action_dim + state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, action_dim),  # 预测对手动作
            )
            for _ in range(num_opponents)
        ])
        
        # 对手类型识别
        self.opponent_type_classifier = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_opponents),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self,
        my_action: torch.Tensor,
        market_state: torch.Tensor,
        opponent_history: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            my_action: [batch, action_dim] 自身动作
            market_state: [batch, state_dim] 市场状态
            opponent_history: [batch, num_opponents, action_dim] 对手历史动作
        Returns:
            predicted_responses: [batch, num_opponents, action_dim]
            opponent_types: [batch, num_opponents] 对手类型概率
        """
        x = torch.cat([my_action, market_state], dim=-1)
        
        # 预测每个对手的响应
        responses = []
        for predictor in self.response_predictors:
            response = predictor(x)
            responses.append(response)
        predicted_responses = torch.stack(responses, dim=1)
        
        # 识别对手类型
        opponent_types = self.opponent_type_classifier(x)
        
        return predicted_responses, opponent_types


class StrategyDenoiser(nn.Module):
    """策略去噪网络：Diffusion 模型的核心"""
    def __init__(
        self,
        action_dim: int,
        time_dim: int = 256,
        condition_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 6
    ):
        super().__init__()
        
        self.action_dim = action_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # 输入投影
        self.input_proj = nn.Linear(action_dim, hidden_dim)
        
        # Transformer 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 条件融合
        self.condition_proj = nn.Linear(condition_dim + time_dim, hidden_dim)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            noisy_action: [batch, action_dim] 带噪声的动作
            timestep: [batch] 时间步
            condition: [batch, condition_dim] 条件嵌入
        Returns:
            predicted_noise: [batch, action_dim] 预测的噪声
        """
        # 时间嵌入
        t_emb = self.time_mlp(timestep)
        
        # 条件嵌入
        cond_emb = self.condition_proj(torch.cat([condition, t_emb], dim=-1))
        
        # 输入投影
        x = self.input_proj(noisy_action)
        
        # 添加条件
        x = x + cond_emb
        
        # Transformer 处理（添加序列维度）
        x = x.unsqueeze(1)  # [batch, 1, hidden_dim]
        x = self.transformer(x)
        x = x.squeeze(1)  # [batch, hidden_dim]
        
        # 输出投影
        predicted_noise = self.output_proj(x)
        
        return predicted_noise


class MarketDiffusionModel(nn.Module):
    """
    完整的市场策略 Diffusion 模型
    整合：条件编码 + 自我影响 + 对手响应 + 策略生成
    """
    def __init__(
        self,
        action_dim: int = 10,          # 动作维度（如：买卖量、价格等）
        state_dim: int = 50,            # 市场状态维度
        opponent_dim: int = 40,         # 对手策略维度
        constraint_dim: int = 20,       # 约束维度
        num_opponents: int = 4,         # 对手数量
        hidden_dim: int = 512,
        num_diffusion_steps: int = 1000,
        beta_schedule: str = 'cosine'
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps
        
        # 条件编码器
        self.condition_encoder = MarketConditionEncoder(
            state_dim=state_dim,
            opponent_dim=opponent_dim,
            constraint_dim=constraint_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=hidden_dim
        )
        
        # 自我影响模块
        self.self_impact = SelfImpactModule(
            action_dim=action_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim // 2
        )
        
        # 对手响应模块
        self.opponent_response = OpponentResponseModule(
            action_dim=action_dim,
            state_dim=state_dim,
            num_opponents=num_opponents,
            hidden_dim=hidden_dim // 2
        )
        
        # 策略去噪器
        self.denoiser = StrategyDenoiser(
            action_dim=action_dim,
            time_dim=256,
            condition_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=6
        )
        
        # Diffusion 参数
        self.register_buffer('betas', self._get_beta_schedule(beta_schedule, num_diffusion_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        
    def _get_beta_schedule(self, schedule: str, num_steps: int) -> torch.Tensor:
        """生成 beta 调度"""
        if schedule == 'linear':
            return torch.linspace(1e-4, 0.02, num_steps)
        elif schedule == 'cosine':
            steps = torch.arange(num_steps + 1, dtype=torch.float32) / num_steps
            alphas_cumprod = torch.cos(((steps + 0.008) / 1.008) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向扩散过程：添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def forward(
        self,
        action: torch.Tensor,
        market_state: torch.Tensor,
        opponent_strategy: torch.Tensor,
        constraints: torch.Tensor,
        timestep: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        训练时的前向传播
        
        Args:
            action: [batch, action_dim] 真实动作
            market_state: [batch, state_dim] 市场状态
            opponent_strategy: [batch, opponent_dim] 对手策略
            constraints: [batch, constraint_dim] 约束
            timestep: [batch] 可选的时间步
        
        Returns:
            outputs: 包含损失和预测的字典
        """
        batch_size = action.shape[0]
        device = action.device
        
        # 随机采样时间步
        if timestep is None:
            timestep = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)
        
        # 生成噪声
        noise = torch.randn_like(action)
        
        # 前向��散
        noisy_action = self.q_sample(action, timestep, noise)
        
        # 编码条件
        condition = self.condition_encoder(market_state, opponent_strategy, constraints)
        
        # 预测噪声
        predicted_noise = self.denoiser(noisy_action, timestep, condition)
        
        # 计算自我影响
        predicted_impact, impact_magnitude = self.self_impact(action, market_state)
        
        # 预测对手响应
        opponent_responses, opponent_types = self.opponent_response(action, market_state)
        
        return {
            'predicted_noise': predicted_noise,
            'true_noise': noise,
            'predicted_impact': predicted_impact,
            'impact_magnitude': impact_magnitude,
            'opponent_responses': opponent_responses,
            'opponent_types': opponent_types,
            'condition': condition
        }
    
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: int,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """单步反向采样"""
        batch_size = x.shape[0]
        device = x.device
        
        # 预测噪声
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        predicted_noise = self.denoiser(x, t_tensor, condition)
        
        # 计算去噪后的 x
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # 均值
        model_mean = (x - (1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.sqrt(alpha_t)
        
        if t == 0:
            return model_mean
        else:
            # 添加噪声
            noise = torch.randn_like(x)
            beta_t = self.betas[t]
            return model_mean + torch.sqrt(beta_t) * noise
    
    @torch.no_grad()
    def sample(
        self,
        market_state: torch.Tensor,
        opponent_strategy: torch.Tensor,
        constraints: torch.Tensor,
        num_samples: int = 1,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        从噪声生成策略（推理）
        
        Args:
            market_state: [batch, state_dim]
            opponent_strategy: [batch, opponent_dim]
            constraints: [batch, constraint_dim]
            num_samples: ���个条件生成的样本数
            return_trajectory: 是否返回完整轨迹
        
        Returns:
            generated_actions: [batch * num_samples, action_dim]
        """
        batch_size = market_state.shape[0]
        device = market_state.device
        
        # 扩展条件
        if num_samples > 1:
            market_state = market_state.repeat_interleave(num_samples, dim=0)
            opponent_strategy = opponent_strategy.repeat_interleave(num_samples, dim=0)
            constraints = constraints.repeat_interleave(num_samples, dim=0)
        
        # 编码条件
        condition = self.condition_encoder(market_state, opponent_strategy, constraints)
        
        # 从纯噪声开始
        x = torch.randn(batch_size * num_samples, self.action_dim, device=device)
        
        trajectory = [x] if return_trajectory else None
        
        # 反向扩散
        for t in reversed(range(self.num_diffusion_steps)):
            x = self.p_sample(x, t, condition)
            if return_trajectory:
                trajectory.append(x)
        
        if return_trajectory:
            return x, torch.stack(trajectory, dim=1)
        return x
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失"""
        # Diffusion 损失（噪声预测）
        diffusion_loss = F.mse_loss(outputs['predicted_noise'], outputs['true_noise'])
        
        # 总损失
        total_loss = diffusion_loss
        
        return {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'impact_magnitude': outputs['impact_magnitude'].mean()
        }


if __name__ == '__main__':
    # 测试
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = MarketDiffusionModel(
        action_dim=10,
        state_dim=50,
        opponent_dim=40,
        constraint_dim=20,
        num_opponents=4
    ).to(device)
    
    # 模拟数据
    batch_size = 8
    action = torch.randn(batch_size, 10).to(device)
    market_state = torch.randn(batch_size, 50).to(device)
    opponent_strategy = torch.randn(batch_size, 40).to(device)
    constraints = torch.randn(batch_size, 20).to(device)
    
    # 前向传播
    outputs = model(action, market_state, opponent_strategy, constraints)
    losses = model.compute_loss(outputs)
    
    print("Model outputs:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
    
    print("\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    # 测试采样
    generated = model.sample(market_state, opponent_strategy, constraints, num_samples=2)
    print(f"\nGenerated actions: {generated.shape}")
    
    print("\n✓ Model test passed!")

