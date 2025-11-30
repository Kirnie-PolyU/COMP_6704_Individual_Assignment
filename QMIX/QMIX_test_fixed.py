import numpy as np
import os
import my_env
from RL_QMIX import QMIX_Trainer, ActionDiscretizer
import matplotlib.pyplot as plt

# Fix OpenMP conflict warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

########################SETTING######################
lane_num = 3
n_veh = 4
width = 120
task_num = 3

IS_TEST = 1

# Get the project root (two levels up from this script's location)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
label = os.path.join(project_root, 'model', 'qmix_model', f'n_{n_veh}')
model_path = os.path.join(label, 'agent')

env = my_env.Environment(lane_num, n_veh, width, task_num)
env.make_new_game()

n_step_per_episode = 100
n_episode_test = 50  # test episodes

#####################################################

def get_agent_states(env):
    """Get individual agent observations"""
    agent_states = []
    for i in range(n_veh):
        task_start = i * task_num
        task_end = (i + 1) * task_num
        agent_obs = np.concatenate([
            env.tk_sac[task_start:task_end],
            [env.d_f[i]],
            [env.ve_l[i]],
            [env.g_channel[i]],
        ])
        agent_states.append(agent_obs)
    
    return np.array(agent_states, dtype=np.float32)


def get_global_state(env):
    """Get global state for mixing network"""
    global_obs = []
    for i in range(n_veh):
        task_start = i * task_num
        task_end = (i + 1) * task_num
        global_obs.extend([
            *env.tk_sac[task_start:task_end],
            env.d_f[i],
            env.ve_l[i],
            env.g_channel[i],
        ])
    return np.array(global_obs, dtype=np.float32)


# ------------------------------------------------------------------------------------------------------------------ #
## Initializations ##
state_dim_per_agent = task_num + 3  # 6
n_actions_per_dim = task_num * 2    # 6

# 计算全局状态维度
sample_global_state = get_global_state(env)
global_state_dim = len(sample_global_state)  # 24

# Action discretization - MUST match training
n_bins = 5
action_discretizer = ActionDiscretizer(
    action_dim=n_actions_per_dim,
    n_bins=n_bins,
    action_min=0.01,
    action_max=1.0
)

# QMIX hyperparameters - MUST match training
hidden_dim = 128
mixing_embed_dim = 32
lr = 5e-4
gamma = 0.99
tau = 0.005

# Initialize QMIX agent
agent = QMIX_Trainer(
    n_agents=n_veh,
    state_dim_per_agent=state_dim_per_agent,
    n_actions_per_agent=action_discretizer.n_actions,
    global_state_dim=global_state_dim,
    hidden_dim=hidden_dim,
    mixing_embed_dim=mixing_embed_dim,
    lr=lr,
    gamma=gamma,
    tau=tau
)

# Load trained model
print(f"\n{'='*70}")
print("QMIX Model Testing")
print(f"{'='*70}")
print(f"Loading model from: {model_path}")
print(f"Model architecture:")
print(f"  - Agents: {n_veh}")
print(f"  - State dim per agent: {state_dim_per_agent}")
print(f"  - Global state dim: {global_state_dim}")
print(f"  - Actions per agent: {action_discretizer.n_actions}")
print(f"  - Hidden dim: {hidden_dim}")
print(f"{'='*70}\n")

try:
    agent.load_model(model_path)
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\n💡 Tip: Make sure you have trained the model first!")
    print("   Run: python QMIX_Train_fixed.py")
    exit(1)

## Let's go
if IS_TEST:
    record_reward_average = []
    t_tk = []
    alloc_tk = []
    
    print("="*70)
    print("Starting QMIX Model Testing...")
    print("="*70 + "\n")
    
    for i_episode in range(n_episode_test):
        if i_episode % 10 == 0:
            print(f"Testing episode: {i_episode}/{n_episode_test}")
        
        record_reward = np.zeros([n_step_per_episode], dtype=np.float32)
        
        # Renew environment periodically
        if i_episode % 20 == 0:
            env.vehicle_renew_position()
            env.renew_channel()
            env.R_V2I()
        
        agent_states_old = get_agent_states(env)
        
        time_tk = 0
        allocation_tk = 0
        
        for i_step in range(n_step_per_episode):
            # Select actions (greedy, epsilon=0)
            discrete_actions = agent.select_actions(agent_states_old, epsilon=0.0)
            
            # Convert to continuous actions
            continuous_actions = action_discretizer.discrete_to_continuous(discrete_actions)
            
            # Prepare action for environment
            action_all_training = np.zeros([n_veh, task_num * 2], dtype=np.float32)
            for i in range(n_veh):
                for j in range(task_num):
                    action_all_training[i, j] = continuous_actions[i, j]
                    action_all_training[i, j + task_num] = continuous_actions[i, j + task_num] * 100
            
            action_channel = action_all_training.copy()
            
            # Take action in environment
            _, train_reward = env.act_for_training(action_channel)
            
            time_tk += np.sum(env.exe_delay) / n_veh
            allocation_tk += np.sum(np.sum(action_channel[:, task_num:], axis=1)) / n_veh
            
            record_reward[i_step] = np.mean(train_reward)
            
            # Get new state
            agent_states_old = get_agent_states(env)
        
        time_tk = time_tk / n_step_per_episode
        alloc_tk.append(allocation_tk / n_step_per_episode)
        t_tk.append(time_tk)
        
        average_reward = np.mean(record_reward)
        record_reward_average.append(average_reward)
    
    # Print test results
    print('\n' + '='*70)
    print('QMIX Test Results:')
    print('='*70)
    print(f'Average Returns: {np.mean(record_reward_average):.4f} ± {np.std(record_reward_average):.4f}')
    print(f'Best Return: {max(record_reward_average):.4f}')
    print(f'Worst Return: {min(record_reward_average):.4f}')
    print(f'Time Consumption for Computing Tasks: {sum(t_tk)/n_episode_test:.6f}')
    print(f'Resources for Computation Tasks: {sum(alloc_tk)/n_episode_test:.4f}')
    print('='*70 + '\n')
    
    # Plot test results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(record_reward_average, linewidth=2, marker='o', markersize=3)
    plt.axhline(y=np.mean(record_reward_average), color='r', linestyle='--',
                label=f'Mean: {np.mean(record_reward_average):.2f}')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('QMIX Test - Episode Rewards', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(t_tk, linewidth=2, marker='s', markersize=3, color='orange')
    plt.axhline(y=sum(t_tk)/n_episode_test, color='r', linestyle='--',
                label=f'Mean: {sum(t_tk)/n_episode_test:.4f}')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Time Consumption', fontsize=12)
    plt.title('Time Consumption per Episode', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(alloc_tk, linewidth=2, marker='^', markersize=3, color='green')
    plt.axhline(y=sum(alloc_tk)/n_episode_test, color='r', linestyle='--',
                label=f'Mean: {sum(alloc_tk)/n_episode_test:.2f}')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Resource Allocation', fontsize=12)
    plt.title('Resource Allocation per Episode', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qmix_test_results.png', dpi=150)
    print("✅ Test plot saved to qmix_test_results.png")
    plt.show()
