#  PPO Algorithm Guide

## 文件说明

### 1. RL_PPO.py
PPO算法的核心实现文件，包含：

- **RolloutBuffer**: 用于存储轨迹数据的缓冲区（替代SAC的ReplayBuffer）
- **ActorNetwork**: 策略网络（Actor），输出动作的均值和标准差
- **CriticNetwork**: 价值网络（Critic），估计状态价值
- **PPO_Trainer**: PPO训练器主类，包含训练和测试的所有功能

### 2. PPO_Train.py
训练脚本，主要功能：
- 初始化环境和PPO agent
- 收集轨迹数据
- 使用PPO算法更新策略
- 保存训练结果和模型

### 3. PPO_test.py
测试脚本，主要功能：
- 加载训练好的模型
- 在测试环境中评估性能
- 输出性能指标（平均奖励、时间消耗、资源分配）

### 4. my_env.txt
环境文件（保持不变）- 定义了车联网通信环境

## SAC vs PPO 主要区别

| 特性 | SAC | PPO |
|------|-----|-----|
| **学习方式** | Off-policy | On-policy |
| **数据效率** | 高（可重复使用历史数据） | 较低（需要最新数据） |
| **稳定性** | 较好 | 非常好 |
| **实现复杂度** | 较高（双Q网络+熵正则化） | 中等 |
| **超参数敏感度** | 较敏感 | 较鲁棒 |
| **适用场景** | 复杂连续控制 | 通用强化学习任务 |

## 关键改进点

### 1. 算法机制
- **SAC**: 使用经验回放缓冲区，通过最大化熵来鼓励探索
- **PPO**: 使用rollout buffer收集在线数据，通过clip机制限制策略更新幅度

### 2. 更新策略
- **SAC**: 每步都可以更新，采样batch_size的历史数据
- **PPO**: 每个episode结束后更新，使用该episode的所有数据，并进行K次epoch更新

### 3. 优势
- **PPO更稳定**: clip机制防止策略更新过大
- **PPO更易调参**: 超参数对性能影响较小
- **PPO更适合**: 需要稳定性和可靠性的实际应用

## 使用方法

### 环境要求
```bash
pip install numpy torch matplotlib scipy
```

### 训练模型
```bash
python PPO_Train.py
```

训练参数：
- `n_episode`: 200 (训练轮数)
- `n_step_per_episode`: 100 (每轮步数)
- `hidden_dim`: 512 (隐藏层维度)
- `lr_actor`: 3e-4 (Actor学习率)
- `lr_critic`: 3e-4 (Critic学习率)
- `gamma`: 0.99 (折扣因子)
- `K_epochs`: 10 (每次更新的epoch数)
- `eps_clip`: 0.2 (PPO clip参数)

### 测试模型
```bash
python PPO_test.py
```

测试参数：
- `n_episode_test`: 50 (测试轮数)
- `DETERMINISTIC`: True (使用确定性策略)

## PPO核心超参数说明

### 1. eps_clip (0.2)
- **作用**: 限制新旧策略比率的变化范围
- **范围**: [1-eps_clip, 1+eps_clip]
- **调参建议**: 
  - 太大：策略更新不稳定
  - 太小：学习速度慢
  - 推荐值：0.1-0.3

### 2. K_epochs (10)
- **作用**: 使用同一批数据更新策略的次数
- **调参建议**:
  - 太大：可能导致过拟合
  - 太小：数据利用不充分
  - 推荐值：5-15

### 3. entropy_coef (0.01)
- **作用**: 鼓励探索的熵正则化系数
- **调参建议**:
  - 太大：策略过于随机
  - 太小：探索不足
  - 推荐值：0.001-0.05

### 4. value_loss_coef (0.5)
- **作用**: 价值函数损失的权重
- **调参建议**:
  - 影响critic的学习速度
  - 推荐值：0.5-1.0

### 5. GAE lambda (0.95)
- **作用**: Generalized Advantage Estimation的折扣参数
- **调参建议**:
  - 接近1：更关注长期回报
  - 接近0：更关注即时回报
  - 推荐值：0.9-0.99

## 模型保存路径

```
model/
└── ppo_model/
    └── n_5/
        ├── agent_actor    # Actor网络参数
        └── agent_critic   # Critic网络参数

data/
└── ppo_data/
    └── n_5/
        ├── ppo_returns.npy      # 训练奖励曲线
        ├── actor_loss.npy       # Actor损失曲线
        ├── critic_loss.npy      # Critic损失曲线
        └── training_results.png # 训练结果图
```

1. **Buffer改变**:
   ```python
   # SAC
   replay_buffer = ReplayBuffer(capacity)
   replay_buffer.push(s, a, r, s', done)
   
   # PPO
   rollout_buffer = RolloutBuffer()
   rollout_buffer.push(s, a, r, s', done, log_prob, value)
   rollout_buffer.clear()  # 每个episode后清空
   ```

2. **动作选择**:
   ```python
   # SAC
   action = agent.policy_net.get_action(state, deterministic)
   
   # PPO
   action, log_prob, value = agent.select_action(state, buffer)
   ```

3. **训练更新**:
   ```python
   # SAC - 每步更新
   if len(replay_buffer) > batch_size:
       agent.update(batch_size)
   
   # PPO - 每个episode更新
   if len(rollout_buffer) > 0:
       agent.update(rollout_buffer)
   ```


