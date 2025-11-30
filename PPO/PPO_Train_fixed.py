import numpy as np
import os
import scipy.io
import my_env
from RL_PPO_fixed import PPO_Trainer, RolloutBuffer
import matplotlib.pyplot as plt

# Fix OpenMP conflict warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

########################SETTING######################
lane_num = 3
n_veh = 5
width = 120
task_num = 3

IS_TRAIN = 1
IS_TEST = 1 - IS_TRAIN

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

label = 'model/ppo_model/n_%d' % n_veh
model_path = label + '/agent'
data_path = 'data/ppo_data/n_%d' % n_veh

# Create directories if they don't exist
os.makedirs(label, exist_ok=True)
os.makedirs(data_path, exist_ok=True)

env = my_env.Environment(lane_num, n_veh, width, task_num)
env.make_new_game()

n_episode = 200
n_step_per_episode = 100
epsi_final = 0.02
epsi_anneal_length = int(0.8 * n_episode)

n_episode_test = 100  # test episodes

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

# PPO hyperparameters - TUNED FOR STABILITY
hidden_dim = 256  # Reduced from 512 for better stability
lr_actor = 1e-4   # Reduced learning rate
lr_critic = 3e-4  # Slightly higher for critic
gamma = 0.99
K_epochs = 5      # Reduced from 10 to prevent overfitting
eps_clip = 0.2
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.5

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
    value_loss_coef=value_loss_coef,
    max_grad_norm=max_grad_norm
)

# Rollout buffer for collecting trajectories
rollout_buffer = RolloutBuffer()

explore_steps = 0  # for random action sampling in the beginning of training
frame_idx = 0

## Let's go
if IS_TRAIN:
    done = 0
    record_reward_average = []
    record_actor_loss = []
    record_critic_loss = []

    for i_episode in range(n_episode):
        print("-------------------------")
        print('Episode:', i_episode)
        
        # Epsilon decay for exploration
        if i_episode < epsi_anneal_length:
            epsi = 1 - i_episode * (1 - epsi_final) / (epsi_anneal_length - 1)
        else:
            epsi = epsi_final
        
        record_reward = np.zeros([n_step_per_episode], dtype=np.float32)  # Changed to float32

        # Renew environment periodically
        if i_episode % 20 == 0:
            env.vehicle_renew_position()
            env.renew_channel()
            env.R_V2I()

        # Clear rollout buffer at the start of each episode
        rollout_buffer.clear()

        state_old_all = []
        state = get_State(env, i_episode / (n_episode - 1), epsi)
        state_old_all.append(state)

        average_reward = 0
        
        # Collect trajectories for one episode
        for i_step in range(n_step_per_episode):
            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_veh, task_num * 2], dtype=np.float32)

            # Select action using PPO policy
            if frame_idx > explore_steps:
                action, log_prob, value = agent.select_action(
                    np.asarray(state_old_all).flatten(), 
                    rollout_buffer, 
                    deterministic=False
                )
            else:
                # Random exploration at the beginning
                action = agent.actor.sample_action(n_output)
                log_prob = 0.0
                value = 0.0
            
            action = np.clip(action, 0.01, 1)
            action_all.append(action)

            # Prepare action for environment
            for i in range(n_veh):
                for j in range(task_num):
                    action_all_training[i, j] = action[j]
                    action_all_training[i, j + task_num] = ((action[j + task_num]) / 3) * 100

            action_channel = action_all_training.copy()
            
            # Take action in environment
            try:
                _, train_reward = env.act_for_training(action_channel)
                
                # Clip reward to prevent extreme values
                train_reward = np.clip(train_reward, -100, 100)
                
                record_reward[i_step] = np.mean(train_reward)
            except Exception as e:
                print(f"Error in environment step: {e}")
                record_reward[i_step] = 0
                train_reward = np.zeros(n_veh)

            # Get new state
            state_new = get_State(env, i_episode / (n_episode - 1), epsi)
            state_new_all.append(state_new)

            # Store transition in rollout buffer
            rollout_buffer.push(
                np.asarray(state_old_all).flatten(),
                np.asarray(action_all).flatten(),
                np.mean(train_reward),
                np.asarray(state_new_all).flatten(),
                done,
                log_prob,
                value
            )

            # Update state
            state_old_all = state_new_all
            frame_idx += 1

        # Update PPO policy after collecting one episode of data
        if len(rollout_buffer) > 0:
            try:
                actor_loss, critic_loss = agent.update(rollout_buffer)
                record_actor_loss.append(actor_loss)
                record_critic_loss.append(critic_loss)
                print(f'Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}')
            except Exception as e:
                print(f"Error during update: {e}")
                record_actor_loss.append(0.0)
                record_critic_loss.append(0.0)

        average_reward = np.mean(record_reward)
        record_reward_average.append(average_reward)
        print('Episode:', i_episode, 'Average Reward:', average_reward)

        # Save model periodically
        if (i_episode + 1) % 50 == 0 and i_episode != 0:
            agent.save_model(model_path)
            print(f"Checkpoint saved at episode {i_episode + 1}")

    # Save final model and results
    agent.save_model(model_path)
    
    # Save training data
    np.save(os.path.join(data_path, 'ppo_returns.npy'), record_reward_average)
    np.save(os.path.join(data_path, 'actor_loss.npy'), record_actor_loss)
    np.save(os.path.join(data_path, 'critic_loss.npy'), record_critic_loss)

    # Plot results
    x = np.linspace(0, n_episode - 1, n_episode, dtype=int)
    y1 = record_reward_average

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(x, y1, linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('PPO Training - Reward', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(record_actor_loss, linewidth=2, color='orange')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Actor Loss', fontsize=12)
    plt.title('Actor Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(record_critic_loss, linewidth=2, color='green')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Critic Loss', fontsize=12)
    plt.title('Critic Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, 'training_results.png'), dpi=150)
    plt.show()

    print('\n' + '='*60)
    print('Training Done. Model saved.')
    print('='*60)
    print(f'Final Average Reward: {record_reward_average[-1]:.4f}')
    print(f'Best Average Reward: {max(record_reward_average):.4f}')
    print('='*60)
