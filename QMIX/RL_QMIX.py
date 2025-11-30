'''
QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning
Paper: https://arxiv.org/abs/1803.11485

Implementation for continuous action space with discretization
Suitable for multi-agent cooperative tasks like vehicular task offloading
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

GPU = False
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f"QMIX using device: {device}")


class ReplayBuffer:
    """Experience Replay Buffer for QMIX"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, actions, reward, next_state, done, global_state=None):
        """
        Store transition
        state: [n_agents, state_dim]
        actions: [n_agents, action_dim] (discrete action indices)
        reward: scalar (team reward)
        next_state: [n_agents, state_dim]
        done: bool
        global_state: [global_state_dim] (optional, for mixing network)
        """
        self.buffer.append((state, actions, reward, next_state, done, global_state))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, actions, reward, next_state, done, global_state = zip(*batch)
        
        return (
            np.array(state),
            np.array(actions),
            np.array(reward),
            np.array(next_state),
            np.array(done),
            np.array(global_state) if global_state[0] is not None else None
        )
    
    def __len__(self):
        return len(self.buffer)


class AgentNetwork(nn.Module):
    """
    Individual agent Q-network
    Each agent has its own Q-network that outputs Q-values for each action
    """
    def __init__(self, state_dim, n_actions, hidden_dim=128):
        super(AgentNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)
        
        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
        
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, state):
        """
        state: [batch_size, state_dim] or [state_dim]
        output: [batch_size, n_actions] or [n_actions]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class HyperNetwork(nn.Module):
    """
    Hypernetwork that generates weights for the mixing network
    Ensures monotonicity by generating positive weights
    """
    def __init__(self, state_dim, n_agents, mixing_embed_dim, hypernet_embed_dim=64):
        super(HyperNetwork, self).__init__()
        
        self.n_agents = n_agents
        self.mixing_embed_dim = mixing_embed_dim
        
        # Hypernetwork for generating weights of first layer
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, n_agents * mixing_embed_dim)
        )
        
        # Hypernetwork for generating bias of first layer
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        
        # Hypernetwork for generating weights of second layer
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, mixing_embed_dim)
        )
        
        # Hypernetwork for generating bias of second layer (state-dependent)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )
    
    def forward(self, state):
        """
        state: [batch_size, state_dim]
        Returns weights and biases for mixing network
        """
        batch_size = state.shape[0]
        
        # Generate weights and biases
        w1 = torch.abs(self.hyper_w1(state))  # Absolute to ensure monotonicity
        w1 = w1.view(batch_size, self.n_agents, self.mixing_embed_dim)
        b1 = self.hyper_b1(state)
        b1 = b1.view(batch_size, 1, self.mixing_embed_dim)
        
        w2 = torch.abs(self.hyper_w2(state))  # Absolute to ensure monotonicity
        w2 = w2.view(batch_size, self.mixing_embed_dim, 1)
        b2 = self.hyper_b2(state)
        b2 = b2.view(batch_size, 1, 1)
        
        return w1, b1, w2, b2


class MixingNetwork(nn.Module):
    """
    Mixing network that combines individual agent Q-values into total Q-value
    Uses monotonic structure to ensure IGM (Individual-Global-Max) property
    """
    def __init__(self, n_agents, state_dim, mixing_embed_dim=32, hypernet_embed_dim=64):
        super(MixingNetwork, self).__init__()
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim
        
        # Hypernetwork to generate mixing network weights
        self.hyper_net = HyperNetwork(state_dim, n_agents, mixing_embed_dim, hypernet_embed_dim)
    
    def forward(self, agent_qs, state):
        """
        Mix individual agent Q-values into team Q-value
        agent_qs: [batch_size, n_agents]
        state: [batch_size, state_dim]
        output: [batch_size, 1]
        """
        batch_size = agent_qs.shape[0]
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)
        
        # Get weights and biases from hypernetwork
        w1, b1, w2, b2 = self.hyper_net(state)
        
        # First layer
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        
        # Second layer
        q_tot = torch.bmm(hidden, w2) + b2
        
        return q_tot.view(batch_size, 1)


class QMIX_Trainer:
    """
    QMIX Trainer for multi-agent reinforcement learning
    Implements centralized training with decentralized execution (CTDE)
    """
    def __init__(self, n_agents, state_dim_per_agent, n_actions_per_agent, 
                 global_state_dim, hidden_dim=128, mixing_embed_dim=32,
                 lr=5e-4, gamma=0.99, tau=0.005):
        
        self.n_agents = n_agents
        self.state_dim_per_agent = state_dim_per_agent
        self.n_actions_per_agent = n_actions_per_agent
        self.global_state_dim = global_state_dim
        self.gamma = gamma
        self.tau = tau
        
        # Create agent networks (one for all agents with parameter sharing)
        self.agent_net = AgentNetwork(state_dim_per_agent, n_actions_per_agent, hidden_dim).to(device)
        self.target_agent_net = AgentNetwork(state_dim_per_agent, n_actions_per_agent, hidden_dim).to(device)
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        
        # Create mixing networks
        self.mixing_net = MixingNetwork(n_agents, global_state_dim, mixing_embed_dim).to(device)
        self.target_mixing_net = MixingNetwork(n_agents, global_state_dim, mixing_embed_dim).to(device)
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
        
        # Optimizer
        self.params = list(self.agent_net.parameters()) + list(self.mixing_net.parameters())
        self.optimizer = optim.Adam(self.params, lr=lr)
        
        print(f"Agent Network: {self.agent_net}")
        print(f"Mixing Network: {self.mixing_net}")
        print(f"Total parameters: {sum(p.numel() for p in self.params)}")
    
    def select_actions(self, states, epsilon=0.0):
        """
        Select actions for all agents using epsilon-greedy
        states: [n_agents, state_dim]
        Returns: [n_agents] discrete action indices
        """
        if random.random() < epsilon:
            # Random actions
            actions = np.random.randint(0, self.n_actions_per_agent, size=self.n_agents)
        else:
            # Greedy actions
            with torch.no_grad():
                states_tensor = torch.FloatTensor(states).to(device)
                q_values = self.agent_net(states_tensor)  # [n_agents, n_actions]
                actions = q_values.argmax(dim=1).cpu().numpy()
        
        return actions
    
    def update(self, replay_buffer, batch_size=32):
        """
        Update QMIX networks using sampled batch
        """
        if len(replay_buffer) < batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones, global_states = replay_buffer.sample(batch_size)
        
        # Convert to tensors
        # states: [batch_size, n_agents, state_dim]
        # actions: [batch_size, n_agents]
        states_tensor = torch.FloatTensor(states).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states_tensor = torch.FloatTensor(next_states).to(device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        # Use provided global state or create from agent states
        if global_states is not None:
            global_states_tensor = torch.FloatTensor(global_states).to(device)
            next_global_states_tensor = global_states_tensor  # Use same for next (will be updated in next iteration)
        else:
            # Fallback: concatenate all agent states
            global_states_tensor = states_tensor.view(batch_size, -1)
            next_global_states_tensor = next_states_tensor.view(batch_size, -1)
        
        # Compute current Q-values
        agent_qs = []
        for i in range(self.n_agents):
            agent_state = states_tensor[:, i, :]
            q_vals = self.agent_net(agent_state)  # [batch_size, n_actions]
            agent_action = actions_tensor[:, i].unsqueeze(1)  # [batch_size, 1]
            chosen_q = q_vals.gather(1, agent_action)  # [batch_size, 1]
            agent_qs.append(chosen_q)
        
        agent_qs = torch.stack(agent_qs, dim=1).squeeze(-1)  # [batch_size, n_agents]
        
        # Mix agent Q-values
        q_tot = self.mixing_net(agent_qs, global_states_tensor)  # [batch_size, 1]
        
        # Compute target Q-values
        with torch.no_grad():
            target_agent_qs = []
            for i in range(self.n_agents):
                next_agent_state = next_states_tensor[:, i, :]
                target_q_vals = self.target_agent_net(next_agent_state)
                max_q = target_q_vals.max(dim=1, keepdim=True)[0]
                target_agent_qs.append(max_q)
            
            target_agent_qs = torch.stack(target_agent_qs, dim=1).squeeze(-1)
            target_q_tot = self.target_mixing_net(target_agent_qs, next_global_states_tensor)
            
            # TD target
            target = rewards_tensor + (1 - dones_tensor) * self.gamma * target_q_tot
        
        # Compute loss
        loss = F.mse_loss(q_tot, target)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.params, 10.0)
        self.optimizer.step()
        
        # Soft update target networks
        self._soft_update()
        
        return loss.item()
    
    def _soft_update(self):
        """Soft update target networks"""
        for target_param, param in zip(self.target_agent_net.parameters(), self.agent_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_mixing_net.parameters(), self.mixing_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_model(self, path):
        """Save model parameters"""
        torch.save({
            'agent_net': self.agent_net.state_dict(),
            'mixing_net': self.mixing_net.state_dict(),
        }, path + '_qmix')
        print(f"Model saved to {path}_qmix")
    
    def load_model(self, path):
        """Load model parameters"""
        checkpoint = torch.load(path + '_qmix', map_location=device)
        self.agent_net.load_state_dict(checkpoint['agent_net'])
        self.mixing_net.load_state_dict(checkpoint['mixing_net'])
        
        # Update target networks
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
        
        self.agent_net.eval()
        self.mixing_net.eval()
        print(f"Model loaded from {path}_qmix")


class ActionDiscretizer:
    """
    Discretize continuous action space for QMIX
    Each action dimension is discretized into n_bins
    """
    def __init__(self, action_dim, n_bins=5, action_min=0.01, action_max=1.0):
        self.action_dim = action_dim
        self.n_bins = n_bins
        self.action_min = action_min
        self.action_max = action_max
        
        # Total number of discrete actions (n_bins for each dimension)
        self.n_actions = n_bins ** action_dim
        
        # Create action mapping
        self._create_action_mapping()
    
    def _create_action_mapping(self):
        """Create mapping from discrete index to continuous action"""
        self.action_map = []
        
        # Create all combinations of discretized actions
        bins = np.linspace(self.action_min, self.action_max, self.n_bins)
        
        if self.action_dim == 1:
            for b in bins:
                self.action_map.append([b])
        elif self.action_dim == 2:
            for b1 in bins:
                for b2 in bins:
                    self.action_map.append([b1, b2])
        elif self.action_dim == 3:
            for b1 in bins:
                for b2 in bins:
                    for b3 in bins:
                        self.action_map.append([b1, b2, b3])
        else:
            # General case for higher dimensions
            from itertools import product
            for combo in product(bins, repeat=self.action_dim):
                self.action_map.append(list(combo))
        
        self.action_map = np.array(self.action_map)
        print(f"Discretized action space: {self.action_dim}D with {self.n_bins} bins per dimension")
        print(f"Total discrete actions: {self.n_actions}")
    
    def discrete_to_continuous(self, discrete_actions):
        """
        Convert discrete action indices to continuous actions
        discrete_actions: [n_agents] or int
        Returns: [n_agents, action_dim] or [action_dim]
        """
        if isinstance(discrete_actions, int):
            return self.action_map[discrete_actions]
        else:
            return self.action_map[discrete_actions]
