import numpy as np
import os
import scipy.io
import my_env
from RL_PPO import PPO_Trainer, RolloutBuffer
import matplotlib.pyplot as plt


# Fix OpenMP conflict warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


########################SETTING######################
lane_num = 3
n_veh = 6
width = 120
task_num = 3

IS_TEST = 1

# Get the project root (two levels up from this script's location)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
label = os.path.join(project_root, 'model', 'ppo_model', f'n_{n_veh}')
model_path = os.path.join(label, 'agent')

env = my_env.Environment(lane_num, n_veh, width, task_num)
env.make_new_game()

n_step_per_episode = 100
n_episode_test = 50  # test episodes

#####################################################

def get_State(env, ind_episode=1., epsi=0.02):
    D_tk = env.tk_sac
    delta_f = env.d_f
    vehicle_l = env.ve_l
    vehicle_G = env.g_channel
    return np.concatenate((D_tk, delta_f, vehicle_l, vehicle_G, [ind_episode, epsi]))


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
## Initializations ##
n_input = len(get_State(env=env))
n_output = task_num * 2
action_range = 1.0

# PPO hyperparameters (same as training)
# hidden_dim = 512
#
hidden_dim = 256
lr_actor = 3e-4
lr_critic = 3e-4
gamma = 0.99
K_epochs = 10
eps_clip = 0.2
entropy_coef = 0.01
value_loss_coef = 0.5

# Initialize PPO agent
agent = PPO_Trainer(
    state_dim=n_input,
    action_dim=n_output,
    hidden_dim=hidden_dim,
    action_range=action_range,
    lr_actor=lr_actor,
    lr_critic=lr_critic,
    gamma=gamma,
    K_epochs=K_epochs,
    eps_clip=eps_clip,
    entropy_coef=entropy_coef,
    value_loss_coef=value_loss_coef
)

# Load trained model
agent.load_model(model_path)

DETERMINISTIC = True  # Use deterministic policy for testing

## Let's go
if IS_TEST:
    record_reward_average = []
    t_tk = []
    alloc_tk = []
    
    print("\nRestoring the PPO model...")
    
    for i_episode in range(n_episode_test):
        record_reward = np.zeros([n_step_per_episode], dtype=np.float16)

        # Renew environment periodically
        if i_episode % 20 == 0:
            env.vehicle_renew_position()
            env.renew_channel()
            env.R_V2I()

        state_old_all = []
        state = get_State(env)
        state_old_all.append(state)

        average_reward = 0
        time_tk = 0
        allocation_tk = 0
        
        for i_step in range(n_step_per_episode):
            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_veh, task_num * 2], dtype=np.float16)
            
            # Get action from PPO policy (deterministic for testing)
            action = agent.actor.get_action(
                np.asarray(state_old_all).flatten(), 
                deterministic=DETERMINISTIC
            )
            action = np.clip(action, 0.01, 1)
            action_all.append(action)
            
            # Prepare action for environment
            for i in range(n_veh):
                for j in range(task_num):
                    action_all_training[i, j] = action[j]
                    action_all_training[i, j + task_num] = ((action[j + task_num]) / 3) * 100
            
            action_channel = action_all_training.copy()
            
            # Take action in environment
            _, train_reward = env.act_for_training(action_channel)
            
            time_tk += np.sum(env.exe_delay) / n_veh
            allocation_tk += np.sum(np.sum(action_channel[:, 3:], axis=1)) / n_veh

            record_reward[i_step] = np.mean(train_reward)
            
            # Get new state
            state_new = get_State(env)
            state_new_all.append(state_new)
            
            # Update state
            state_old_all = state_new_all
        
        time_tk = time_tk / n_step_per_episode
        alloc_tk.append(allocation_tk / n_step_per_episode)
        t_tk.append(time_tk)
        
        average_reward = np.mean(record_reward)
        record_reward_average.append(average_reward)
    
    # Print test results
    print('\n' + '='*50)
    print('PPO Test Results:')
    print('='*50)
    print(f'Average Returns: {np.mean(record_reward_average):.4f}')
    print(f'Time Consumption for Computing Tasks: {sum(t_tk)/n_episode_test:.6f}')
    print(f'Resources for Computation Tasks: {sum(alloc_tk)/n_episode_test:.4f}')
    print('='*50)
    
    # Plot test results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(record_reward_average)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Test Episode Rewards')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(t_tk)
    plt.xlabel('Episode')
    plt.ylabel('Time Consumption')
    plt.title('Time Consumption per Episode')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(alloc_tk)
    plt.xlabel('Episode')
    plt.ylabel('Resource Allocation')
    plt.title('Resource Allocation per Episode')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(label, 'test_results.png'))
    plt.show()
