# COMP_6704_Individual_Assignment

# Multi-Agent Reinforcement Learning for Vehicular Task Offloading

A comprehensive implementation and comparison of four reinforcement learning algorithms (MARL, SAC, PPO, QMIX) for multi-agent vehicular edge computing task offloading scenarios.

📺 Video Demonstration Link
https://www.youtube.com/watch?v=GPE9PMVB9os

## 🎯 Overview

This project implements and compares four state-of-the-art reinforcement learning algorithms for solving the multi-agent vehicular task offloading problem in edge computing scenarios. The system optimizes task offloading decisions and resource allocation for multiple vehicles to minimize latency and maximize system efficiency.

### Problem Description

- **Scenario**: Multiple vehicles (n_veh=4) need to offload computation tasks to edge servers
- **Objective**: Minimize task completion time while optimizing resource allocation
- **Challenges**: 
  - Multi-agent coordination
  - Dynamic wireless channels
  - Limited computational resources
  - Real-time decision making
### Key Features

✅ **Four Algorithm Implementations**: MARL, SAC, PPO, QMIX  
✅ **Comprehensive Comparison**: Performance metrics and analysis  
✅ **Well-Documented**: Detailed usage guides and API documentation  
✅ **Ready to Use**: Pre-configured hyperparameters and training scripts  
✅ **Extensible**: Modular design for easy customization  

---
## 🤖 Algorithms Implemented

### 1. MARL (Multi-Agent Reinforcement Learning)
- **Type**: Custom multi-agent algorithm
- **Approach**: Distributed learning with shared experience
- **Best For**: Baseline comparison, understanding multi-agent dynamics

### 2. SAC (Soft Actor-Critic)
- **Type**: Off-policy, Actor-Critic
- **Action Space**: Continuous
- **Key Features**:
  - Entropy regularization for exploration
  - Twin Q-networks to reduce overestimation
  - Automatic temperature tuning
- **Best For**: Single-agent optimization, fastest convergence

**Performance**: Average Reward = **13.52**

### 3. PPO (Proximal Policy Optimization)
- **Type**: On-policy, Policy Gradient
- **Action Space**: Continuous/Discrete
- **Key Features**:
  - Clipped surrogate objective for stable updates
  - Generalized Advantage Estimation (GAE)
  - Multiple epochs per batch
- **Best For**: Stable training, easy hyperparameter tuning

**Performance**: Average Reward = **12.02**

### 4. QMIX (Monotonic Value Function Factorisation) ⭐
- **Type**: Off-policy, Value-based, Multi-agent
- **Action Space**: Discrete
- **Key Features**:
  - Centralized Training, Decentralized Execution (CTDE)
  - Monotonicity constraint for global optimality
  - Individual agent networks + mixing network
- **Best For**: Multi-agent cooperation, scalability

**Performance**: Average Reward = **14.18** (BEST)

## 🚀 Quick Start

### Option 1: QMIX (Recommended for Multi-Agent Scenarios)

```bash
# 1. Navigate to project directory
cd your_project/

# 2. Train QMIX model
python QMIX_Train_fixed.py

# Expected training time: ~50-60 minutes (300 episodes)

# 3. Test trained model
python QMIX_test_fixed.py
```

### Option 2: SAC (Fastest Convergence)

```bash
# Train SAC model
python SAC_Train.py

# Test trained model
python SAC_test.py
```

### Option 3: PPO (Most Stable)

```bash
# Train PPO model (use fixed version)
python PPO_Train_fixed.py

# Test trained model
python PPO_test_fixed.py
```

### Option 4: MARL (Baseline)

```bash
# Train MARL model
python marl_Train.py

# Test trained model
python algorithm_test.py
```

---
## 📖 Detailed Usage

### Training Configuration

All algorithms share similar configuration parameters:

```python
# Environment Configuration
lane_num = 3        # Number of lanes
n_veh = 4           # Number of vehicles
width = 120         # Scenario width
task_num = 3        # Tasks per vehicle

# Training Configuration
n_episode = 300                # Training episodes
n_step_per_episode = 100       # Steps per episode
batch_size = 32                # Batch size
learning_rate = 5e-4           # Learning rate
gamma = 0.99                   # Discount factor
```

### QMIX-Specific Configuration

```python
# Action Discretization
n_bins = 5                     # Discretization bins per dimension
action_min = 0.01              # Minimum action value
action_max = 1.0               # Maximum action value

# Network Architecture
hidden_dim = 128               # Agent network hidden dimension
mixing_embed_dim = 32          # Mixing network embedding dimension

# Exploration
epsilon_start = 1.0            # Initial exploration rate
epsilon_final = 0.05           # Final exploration rate
epsilon_anneal_length = 210    # Annealing episodes (0.7 * n_episode)
```

### Customizing Hyperparameters

Edit the configuration section in training scripts:

**For QMIX** (`QMIX_Train_fixed.py`):
```python
# Line 30-40: Adjust training parameters
n_episode = 300                # Increase for better convergence
epsilon_final = 0.05           # Lower for less exploration

# Line 86-90: Adjust discretization
n_bins = 3                     # Reduce for faster training
                               # (729 actions instead of 15625)

# Line 95-99: Adjust network size
hidden_dim = 256               # Increase for more capacity
lr = 1e-4                      # Decrease for stability
```

**For PPO** (`PPO_Train_fixed.py`):
```python
# Adjust PPO-specific parameters
eps_clip = 0.2                 # Clip range
K_epochs = 10                  # Update epochs per batch
gae_lambda = 0.95              # GAE parameter
```

**For SAC** (`SAC_Train.py`):
```python
# Adjust SAC-specific parameters
alpha = 0.2                    # Entropy coefficient
tau = 0.005                    # Soft update coefficient
buffer_size = 200000           # Replay buffer size
```

---
## ⚙️ Configuration

### Environment Parameters

The environment is defined in `my_env.py`:

```python
class Environment:
    def __init__(self, lane_num, n_veh, width, task_num):
        self.lane_num = lane_num       # Number of lanes
        self.n_veh = n_veh             # Number of vehicles
        self.width = width             # Scenario width (meters)
        self.task_num = task_num       # Tasks per vehicle
        
        # Communication parameters
        self.bandwidth = 20e6          # Channel bandwidth (Hz)
        self.power = 0.1               # Transmission power (W)
        self.noise = 1e-13             # Noise power (W)
        
        # Computation parameters
        self.cpu_freq_local = 1e9      # Local CPU frequency (Hz)
        self.cpu_freq_edge = 5e9       # Edge CPU frequency (Hz)
```

### State Space

**Agent Observation** (per vehicle):
- Task requirements: `[task_1, task_2, task_3]` (3 dimensions)
- Frequency allocation: `delta_f` (1 dimension)
- Vehicle position: `position` (1 dimension)
- Channel gain: `channel_gain` (1 dimension)

**Total per agent**: 6 dimensions

**Global State** (for QMIX):
- Concatenation of all agent observations
- **Total**: 4 vehicles × 6 dimensions = 24 dimensions

### Action Space

**Continuous Version** (SAC, PPO, MARL):
- Offloading decisions: `[o_1, o_2, o_3]` ∈ [0, 1]
- Resource allocation: `[r_1, r_2, r_3]` ∈ [0, 1]
- **Total per agent**: 6 dimensions

**Discrete Version** (QMIX):
- Each continuous action discretized into `n_bins` values
- **Total discrete actions**: n_bins^6 = 5^6 = 15,625

### Reward Function

```python
reward = -α * latency - β * energy_consumption + γ * task_completion
```

Where:
- `latency`: Total task completion time
- `energy_consumption`: Communication and computation energy
- `task_completion`: Number of successfully completed tasks

---

## 📈 Results

### Trained Models

Trained models are saved in the following directories:

```
model/
├── marl_model/n_4/           # MARL trained model
├── sac_model/n_4/            # SAC trained model
├── ppo_model/n_4/            # PPO trained model
└── qmix_model/n_4/           # QMIX trained model
    └── agent_qmix            # QMIX model file
```
### Training Data

Training metrics are saved as NumPy arrays:

```
data/
├── marl_data/n_4/
│   ├── marl_returns.npy      # Episode rewards
│   └── marl_loss.npy         # Training losses
├── sac_data/n_4/
│   ├── sac_returns.npy
│   └── sac_loss.npy
├── ppo_data/n_4/
│   ├── ppo_returns.npy
│   └── ppo_loss.npy
└── qmix_data/n_4/
    ├── qmix_returns.npy
    ├── qmix_loss.npy
    └── training_results.png   # Training curves
```

## 🐛 Troubleshooting

### Common Issues

#### Issue 1: Dimension Mismatch Error (QMIX)

**Error**:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x24 and 26x64)
```

**Solution**:
- Use the **fixed version**: `QMIX_Train_fixed.py` and `QMIX_test_fixed.py`
- See `QMIX_BUGFIX.md` for detailed explanation

#### Issue 2: NaN Loss During Training (PPO)

**Error**:
```
ValueError: nan values in loc parameter
```

**Solution**:
- Use the **fixed version**: `PPO_Train_fixed.py`
- See `BUGFIX_README.md` for 10 critical fixes

#### Issue 3: Model Loading Failure

**Error**:
```
Error loading model: size mismatch
```

**Solution**:
Ensure training and testing configurations match:
```python
# These parameters must be identical
n_veh = 4
task_num = 3
hidden_dim = 128      # QMIX/PPO
n_bins = 5            # QMIX only
```

#### Issue 4: Slow Training

**Problem**: Training takes too long

**Solutions**:

1. **Reduce action space** (QMIX):
```python
n_bins = 3  # From 5 to 3 (729 actions instead of 15625)
```

2. **Use GPU**:
```python
# In RL_*.py files
GPU = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

3. **Reduce episodes**:
```python
n_episode = 200  # From 300 to 200
```

#### Issue 5: Low Performance

**Problem**: Model performance below expected

**Solutions**:

1. **Increase training episodes**:
```python
n_episode = 500
```

2. **Adjust exploration** (QMIX):
```python
epsilon_final = 0.1           # More exploration
epsilon_anneal_length = 400   # Longer exploration period
```

3. **Tune learning rate**:
```python
lr = 1e-4  # Lower for more stable learning
```

4. **Increase network capacity**:
```python
hidden_dim = 256  # From 128 to 256
```

---
## 📚 References

### Algorithm Papers

1. **QMIX**: Rashid, T., et al. (2018). "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning." *ICML 2018*.
   - Paper: https://arxiv.org/abs/1803.11485

2. **SAC**: Haarnoja, T., et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." *ICML 2018*.
   - Paper: https://arxiv.org/abs/1801.01290

3. **PPO**: Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv preprint*.
   - Paper: https://arxiv.org/abs/1707.06347

4. **MARL**: Lowe, R., et al. (2017). "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." *NIPS 2017*.
   - Paper: https://arxiv.org/abs/1706.02275

### Related Work

- Value Decomposition Networks (VDN)
- QTRAN: Learning to Factorize with Transformation
- QPLEX: Duplex Dueling Multi-Agent Q-Learning
---

## 📄 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Performance improvements
- New algorithm implementations
- Documentation enhancements

---

## 📧 Contact

For questions, issues, or collaborations, please:
- Open an issue on GitHub
- Check the documentation files in each algorithm folder
- Refer to the troubleshooting guides

---
**Happy Training!** 🎉

