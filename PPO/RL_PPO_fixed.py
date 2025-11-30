'''
Proximal Policy Optimization (PPO) - Fixed Version
Implementation for continuous action space with numerical stability improvements
Paper: https://arxiv.org/abs/1707.06347
'''

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt

GPU = False
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


class RolloutBuffer:
    """Buffer for storing trajectories experienced by PPO agent"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def push(self, state, action, reward, next_state, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def get(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.next_states),
            np.array(self.dones),
            np.array(self.log_probs),
            np.array(self.values)
        )
    
    def __len__(self):
        return len(self.states)


class ActorNetwork(nn.Module):
    """Policy Network (Actor) for continuous action space with improved stability"""
    def __init__(self, state_dim, action_dim, hidden_dim, action_range=1.0, 
                 init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(ActorNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_range = action_range
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Weight initialization with orthogonal init for better stability
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)
        
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.constant_(self.mean.bias, 0)
        nn.init.constant_(self.log_std.bias, 0)
    
    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        
        mean = torch.sigmoid(self.mean(x))  # Use sigmoid to constrain output to [0, 1]
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def get_action(self, state, deterministic=False):
        """Get action for interaction with environment"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            action = mean
        else:
            normal = Normal(mean, std)
            action = normal.sample()
        
        # Clip action to valid range
        action = torch.clamp(action, 0.01, self.action_range)
        
        return action.detach().cpu().numpy()[0]
    
    def evaluate(self, state, action):
        """Evaluate log probability and entropy for given state-action pairs"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Add small epsilon to prevent numerical issues
        std = torch.clamp(std, min=1e-6)
        
        normal = Normal(mean, std)
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Clamp log_prob to prevent extreme values
        log_prob = torch.clamp(log_prob, min=-20, max=2)
        
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy
    
    def sample_action(self, action_dim):
        """Sample random action"""
        action = torch.FloatTensor(action_dim).uniform_(0.01, 1.0)
        return action.numpy()


class CriticNetwork(nn.Module):
    """Value Network (Critic) with improved stability"""
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
        # Orthogonal initialization for better stability
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.constant_(self.value.bias, 0)
    
    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        value = self.value(x)
        return value


class PPO_Trainer:
    def __init__(self, state_dim, action_dim, hidden_dim=512, action_range=1.0,
                 lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, K_epochs=10, 
                 eps_clip=0.2, entropy_coef=0.01, value_loss_coef=0.5,
                 max_grad_norm=0.5):
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        
        # Actor-Critic networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim, action_range).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(device)
        
        # Create old policy for computing ratio
        self.actor_old = ActorNetwork(state_dim, action_dim, hidden_dim, action_range).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        # Optimizers with weight decay for regularization
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)
        
        # Loss function for critic with Huber loss for robustness
        self.MseLoss = nn.SmoothL1Loss()  # More robust than MSE
        
        print('Actor Network:', self.actor)
        print('Critic Network:', self.critic)
    
    def select_action(self, state, rollout_buffer, deterministic=False):
        """Select action and store in buffer (for training)"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mean, log_std = self.actor_old.forward(state_tensor)
            std = log_std.exp()
            std = torch.clamp(std, min=1e-6)
            
            if deterministic:
                action = mean
            else:
                normal = Normal(mean, std)
                action = normal.sample()
            
            # Clip action
            action = torch.clamp(action, 0.01, 1.0)
            
            # Calculate log probability
            log_prob = Normal(mean, std).log_prob(action).sum(dim=-1)
            log_prob = torch.clamp(log_prob, min=-20, max=2)
            
            # Get state value
            value = self.critic(state_tensor)
        
        action_np = action.cpu().numpy()[0]
        
        return action_np, log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = []
        gae = 0
        
        # Normalize rewards to prevent extreme values
        rewards = np.array(rewards)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = next_values[i]
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + gamma * lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, rollout_buffer, batch_size=64):
        """Update policy using PPO algorithm with improved numerical stability"""
        
        # Get data from buffer
        states, actions, rewards, next_states, dones, old_log_probs, values = rollout_buffer.get()
        
        # Check for NaN in input data
        if np.isnan(states).any() or np.isnan(actions).any() or np.isnan(rewards).any():
            print("Warning: NaN detected in buffer data, skipping update")
            return 0.0, 0.0
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        
        # Compute next values for GAE
        with torch.no_grad():
            next_values = self.critic(torch.FloatTensor(next_states).to(device)).cpu().numpy().flatten()
        
        # Compute advantages using GAE
        advantages, returns = self.compute_gae(rewards, values, next_values, dones, self.gamma)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Clip advantages to prevent extreme values
        advantages = np.clip(advantages, -10, 10)
        
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        
        # Optimize policy for K epochs
        actor_losses = []
        critic_losses = []
        
        for epoch in range(self.K_epochs):
            # Check for NaN in network parameters
            if torch.isnan(list(self.actor.parameters())[0]).any():
                print("Warning: NaN detected in actor parameters, stopping update")
                break
            
            # Get current policy log probabilities and entropy
            log_probs, entropy = self.actor.evaluate(states, actions)
            
            # Get current state values
            state_values = self.critic(states).squeeze()
            
            # Calculate ratios (pi_theta / pi_theta_old)
            ratios = torch.exp(log_probs.squeeze() - old_log_probs)
            
            # Clip ratios to prevent extreme values
            ratios = torch.clamp(ratios, 0.1, 10.0)
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Calculate actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values, returns)
            entropy_loss = -entropy.mean()
            
            # Check for NaN in losses
            if torch.isnan(actor_loss) or torch.isnan(critic_loss):
                print(f"Warning: NaN detected in loss at epoch {epoch}, stopping update")
                break
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            
            # Update actor
            self.optimizer_actor.zero_grad()
            actor_loss_backward = actor_loss + self.entropy_coef * entropy_loss
            actor_loss_backward.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer_actor.step()
            
            # Update critic
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer_critic.step()
        
        # Copy new weights into old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        
        return avg_actor_loss, avg_critic_loss
    
    def save_model(self, path):
        """Save model parameters"""
        torch.save(self.actor.state_dict(), path + '_actor')
        torch.save(self.critic.state_dict(), path + '_critic')
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model parameters"""
        self.actor.load_state_dict(torch.load(path + '_actor', map_location=device))
        self.critic.load_state_dict(torch.load(path + '_critic', map_location=device))
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        self.actor.eval()
        self.critic.eval()
        self.actor_old.eval()
        print(f"Model loaded from {path}")
