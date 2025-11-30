import numpy as np
import os
import my_env
from RL_QMIX import QMIX_Trainer, ReplayBuffer, ActionDiscretizer
import matplotlib.pyplot as plt

# Fix OpenMP conflict warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

########################SETTING######################
lane_num = 3
n_veh = 4
width = 120
task_num = 3

IS_TRAIN = 1
IS_TEST = 1 - IS_TRAIN

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

label = 'model/qmix_model/n_%d' % n_veh
model_path = label + '/agent'
data_path = 'data/qmix_data/n_%d' % n_veh

# Create directories if they don't exist
os.makedirs(label, exist_ok=True)
os.makedirs(data_path, exist_ok=True)

env = my_env.Environment(lane_num, n_veh, width, task_num)
env.make_new_game()

n_episode = 300
n_step_per_episode = 100
epsilon_start = 1.0
epsilon_final = 0.05
epsilon_anneal_length = int(0.7 * n_episode)

batch_size = 32
buffer_size = 10000
update_interval = 1  # Update every step

#####################################################

def get_global_state(env):
    """
    Get global state for mixing network
    Global state includes all vehicles' information
    """
    # 修复：使用简化的全局状态
    global_obs = []
    for i in range(n_veh):
        task_start = i * task_num
        task_end = (i + 1) * task_num
        global_obs.extend([
            *env.tk_sac[task_start:task_end],  # 车辆i的任务 (3维)
            env.d_f[i],                         # 车辆i的delta_f (1维)
            env.ve_l[i],                        # 车辆i的位置 (1维)
            env.g_channel[i],                   # 车辆i的信道增益 (1维)
        ])
    # 总维度：n_veh * (task_num + 3) = 4 * (3 + 3) = 24
    return np.array(global_obs, dtype=np.float32)


def get_agent_states(env):
    """
    Get individual agent observations
    Each agent observes: its own tasks, position, channel gain, delta_f
    """
    agent_states = []
    for i in range(n_veh):
        task_start = i * task_num
        task_end = (i + 1) * task_num
        agent_obs = np.concatenate([
            env.tk_sac[task_start:task_end],  # Own tasks (3维)
            [env.d_f[i]],                      # Own delta_f (1维)
            [env.ve_l[i]],                     # Own position (1维)
            [env.g_channel[i]],                # Own channel gain (1维)
        ])
        agent_states.append(agent_obs)
    
    return np.array(agent_states, dtype=np.float32)  # [n_veh, 6]


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
## Initializations ##

# State dimensions
state_dim_per_agent = task_num + 3  # tasks + delta_f + position + channel = 3+3 = 6
n_actions_per_dim = task_num * 2    # offloading decision + resource allocation = 6

# 修复：正确计算全局状态维度
# 初始化环境获取真实维度
sample_global_state = get_global_state(env)
global_state_dim = len(sample_global_state)  # 应该是 4 * 6 = 24

print(f"\n{'='*70}")
print(f"Environment Configuration:")
print(f"{'='*70}")
print(f"Vehicles: {n_veh}")
print(f"Tasks per vehicle: {task_num}")
print(f"Agent observation dim: {state_dim_per_agent}")
print(f"Global state dim: {global_state_dim}")
print(f"Action dim per agent: {n_actions_per_dim}")
print(f"{'='*70}\n")

# Action discretization
# Each agent has task_num*2 continuous actions
# Discretize each action dimension into n_bins
n_bins = 5  # 5 discrete values per action dimension
action_discretizer = ActionDiscretizer(
    action_dim=n_actions_per_dim,
    n_bins=n_bins,
    action_min=0.01,
    action_max=1.0
)

print(f"Action discretization:")
print(f"  - Bins per dimension: {n_bins}")
print(f"  - Action dimensions: {n_actions_per_dim}")
print(f"  - Total discrete actions: {action_discretizer.n_actions}")
print(f"{'='*70}\n")

# QMIX hyperparameters
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

# Replay buffer
replay_buffer = ReplayBuffer(buffer_size)

## Let's go
if IS_TRAIN:
    record_reward_average = []
    record_loss = []
    
    print("\n" + "="*70)
    print("QMIX Training Started")
    print("="*70)
    print(f"Episodes: {n_episode}")
    print(f"Steps per episode: {n_step_per_episode}")
    print(f"Agents: {n_veh}")
    print(f"Discrete actions per agent: {action_discretizer.n_actions}")
    print(f"Action discretization: {n_bins} bins per dimension")
    print(f"Global state dim: {global_state_dim}")
    print("="*70 + "\n")
    
    total_steps = 0
    
    for i_episode in range(n_episode):
        print("-------------------------")
        print('Episode:', i_episode)
        
        # Epsilon decay
        if i_episode < epsilon_anneal_length:
            epsilon = epsilon_start - (epsilon_start - epsilon_final) * (i_episode / epsilon_anneal_length)
        else:
            epsilon = epsilon_final
        
        record_reward = np.zeros([n_step_per_episode], dtype=np.float32)
        episode_loss = []
        
        # Renew environment periodically
        if i_episode % 20 == 0:
            env.vehicle_renew_position()
            env.renew_channel()
            env.R_V2I()
        
        # Get initial agent states
        agent_states_old = get_agent_states(env)
        
        for i_step in range(n_step_per_episode):
            # Select actions for all agents
            discrete_actions = agent.select_actions(agent_states_old, epsilon=epsilon)
            
            # Convert discrete actions to continuous
            continuous_actions = action_discretizer.discrete_to_continuous(discrete_actions)
            
            # Prepare action for environment
            action_all_training = np.zeros([n_veh, task_num * 2], dtype=np.float32)
            for i in range(n_veh):
                for j in range(task_num):
                    # Offloading decision
                    action_all_training[i, j] = continuous_actions[i, j]
                    # Resource allocation (scale to 0-100)
                    action_all_training[i, j + task_num] = continuous_actions[i, j + task_num] * 100
            
            action_channel = action_all_training.copy()
            
            # Take action in environment
            try:
                _, train_reward = env.act_for_training(action_channel)
                train_reward = np.clip(train_reward, -100, 100)
                team_reward = np.mean(train_reward)  # Use mean reward as team reward
                record_reward[i_step] = team_reward
            except Exception as e:
                print(f"Error in environment step: {e}")
                record_reward[i_step] = 0
                team_reward = 0
            
            # Get new agent states and global state
            agent_states_new = get_agent_states(env)
            global_state = get_global_state(env)
            
            # Store transition in replay buffer
            done = False  # Episode termination
            replay_buffer.push(
                agent_states_old,
                discrete_actions,
                team_reward,
                agent_states_new,
                done,
                global_state  # 添加全局状态
            )
            
            # Update QMIX
            if total_steps % update_interval == 0 and len(replay_buffer) >= batch_size:
                loss = agent.update(replay_buffer, batch_size)
                episode_loss.append(loss)
            
            # Update state
            agent_states_old = agent_states_new
            total_steps += 1
        
        average_reward = np.mean(record_reward)
        average_loss = np.mean(episode_loss) if episode_loss else 0.0
        
        record_reward_average.append(average_reward)
        record_loss.append(average_loss)
        
        print(f'Epsilon: {epsilon:.3f}, Avg Reward: {average_reward:.4f}, Loss: {average_loss:.4f}')
        
        # Save model periodically
        if (i_episode + 1) % 50 == 0 and i_episode != 0:
            agent.save_model(model_path)
            print(f"✅ Checkpoint saved at episode {i_episode + 1}")
    
    # Save final model and results
    agent.save_model(model_path)
    print(f"\n✅ Final model saved to {model_path}")
    
    # Save training data
    np.save(os.path.join(data_path, 'qmix_returns.npy'), record_reward_average)
    np.save(os.path.join(data_path, 'qmix_loss.npy'), record_loss)
    
    # Plot results
    x = np.linspace(0, n_episode - 1, n_episode, dtype=int)
    y1 = record_reward_average
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, y1, linewidth=2, color='blue')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('QMIX Training - Reward', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(record_loss, linewidth=2, color='red')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('QMIX Training - Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, 'training_results.png'), dpi=150)
    print(f"✅ Training plot saved to {os.path.join(data_path, 'training_results.png')}")
    plt.show()
    
    print("\n" + "="*70)
    print("Training Completed!")
    print("="*70)
    print(f"Final Average Reward: {record_reward_average[-1]:.4f}")
    print(f"Best Average Reward: {max(record_reward_average):.4f}")
    print("="*70)
