'''
Proximal Policy Optimization (PPO)
Implementation for continuous action space
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
    """Policy Network (Actor) for continuous action space"""
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
        
        # Weight initialization
        self.mean.weight.data.uniform_(-init_w, init_w)
        self.mean.bias.data.uniform_(-init_w, init_w)
        self.log_std.weight.data.uniform_(-init_w, init_w)
        self.log_std.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = self.mean(x)
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
        
        normal = Normal(mean, std)
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy
    
    def sample_action(self, action_dim):
        """Sample random action"""
        action = torch.FloatTensor(action_dim).uniform_(0.01, 1.0)
        return action.numpy()


class CriticNetwork(nn.Module):
    """Value Network (Critic)"""
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
        # Weight initialization
        self.value.weight.data.uniform_(-init_w, init_w)
        self.value.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.value(x)
        return value


class PPO_Trainer:
    def __init__(self, state_dim, action_dim, hidden_dim=512, action_range=1.0,
                 lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, K_epochs=10, 
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
        
        # Optimizers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Loss function for critic
        self.MseLoss = nn.MSELoss()
        
        print('Actor Network:', self.actor)
        print('Critic Network:', self.critic)
    
    def select_action(self, state, rollout_buffer, deterministic=False):
        """Select action and store in buffer (for training)"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mean, log_std = self.actor_old.forward(state_tensor)
            std = log_std.exp()
            
            if deterministic:
                action = mean
            else:
                normal = Normal(mean, std)
                action = normal.sample()
            
            # Clip action
            action = torch.clamp(action, 0.01, 1.0)
            
            # Calculate log probability
            log_prob = Normal(mean, std).log_prob(action).sum(dim=-1)
            
            # Get state value
            value = self.critic(state_tensor)
        
        action_np = action.cpu().numpy()[0]
        
        return action_np, log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = []
        gae = 0
        
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
        """Update policy using PPO algorithm"""
        
        # Get data from buffer
        states, actions, rewards, next_states, dones, old_log_probs, values = rollout_buffer.get()
        
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
        
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        
        # Optimize policy for K epochs
        for epoch in range(self.K_epochs):
            # Get current policy log probabilities and entropy
            log_probs, entropy = self.actor.evaluate(states, actions)
            
            # Get current state values
            state_values = self.critic(states).squeeze()
            
            # Calculate ratios (pi_theta / pi_theta_old)
            ratios = torch.exp(log_probs.squeeze() - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Calculate actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values, returns)
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = actor_loss + self.value_loss_coef * critic_loss + self.entropy_coef * entropy_loss
            
            # Update actor
            self.optimizer_actor.zero_grad()
            actor_loss_backward = actor_loss + self.entropy_coef * entropy_loss
            actor_loss_backward.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer_actor.step()
            
            # Update critic
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer_critic.step()
        
        # Copy new weights into old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        return actor_loss.item(), critic_loss.item()
    
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
