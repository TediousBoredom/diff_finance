# DiffFinance: AI + Game Theory Market Participation Modeling

基于 Diffusion 模型的对抗式市场参与建模系统，构建能够感知自身决策影响、他方策略响应与制度约束的 AI 参与模型。

## 核心特性

### 1. 博弈论框架
- **多智能体建模**：支持多个市场参与者（做市商、投机者、套利者）
- **纳什均衡求解**：基于 Diffusion 的均衡策略生成
- **动态博弈**：时序决策与策略演化

### 2. Diffusion 架构
- **策略扩散模型**：生成最优交易策略
- **价格轨迹生成**：预测市场价格演化
- **条件生成**：基于市场状态、对手策略、制度约束的条件生成

### 3. 自我意识机制
- **影响力建模**：量化自身订单对价格的冲击
- **足迹追踪**：识别自身历史行为的市场痕迹
- **反身性建模**：考虑预期的自我实现效应

### 4. 策略响应预测
- **对手建模**：推断其他参与者的目标函数
- **反应函数学习**：预测对手对自身行动的响应
- **高阶信念**：建模"我认为他认为我会..."的递归推理

### 5. 制度约束
- **监管规则**：涨跌停、熔断机制、持仓限制
- **市场微观结构**：订单簿机制、撮合规则
- **交易成本**：手续���、滑点、市场冲击

## 系统架构

```
difffinance/
├── models/
│   ├── diffusion/          # Diffusion 核心模型
│   ├── game_theory/        # 博弈论模块
│   ├── market/             # 市场环境模拟
│   └── agents/             # 智能体定义
├── training/               # 训练脚本
├── inference/              # 推理与策略生成
├── evaluation/             # 评估与回测
└── configs/                # 配置文件
```

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 训练模型
```bash
python training/train_market_diffusion.py --config configs/default.yaml
```

### 运行市场模拟
```bash
python inference/simulate_market.py --agents 5 --episodes 100
```

### 评估策略
```bash
python evaluation/evaluate_strategies.py --checkpoint checkpoints/best_model.pt
```

## 理论基础

### Diffusion + Game Theory
将市场策略视为从噪声到最优策略的扩散过程：
- **前向过程**：最优策略 → 随机噪声
- **反向过程**：噪声 → 最优策略（通过去噪学习）
- **条件生成**：给定市场状态 s、对手策略 a_{-i}、约束 c，生成最优响应 a_i*

### 数学框架
```
目标：max E[U_i(a_i, a_{-i}, s) | constraints]
约束：
  - 自我影响：p_{t+1} = f(p_t, a_i, impact_i)
  - 对手响应：a_{-i} ~ π_{-i}(s, a_i)
  - 制度约束：g(a_i, s) ≤ 0
```

## 应用场景

1. **算法交易**：高频交易策略优化
2. **做市商**：库存管理与报价策略
3. **风险管理**：对抗式压力测试
4. **监管科技**：市场操纵检测
5. **机制设计**：交易规则优化

## 引用

如果使用本项目，请引用：
```bibtex
@software{difffinance2025,
  title={DiffFinance: Diffusion-based Adversarial Market Participation Modeling},
  author={Your Name},
  year={2025}
}
```

## License

MIT License

