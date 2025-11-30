import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Load data (assuming paths are correct)
i = 4
maddpg_data = np.load('model/marl_model/marl_n_%d/returns.pkl.npy' % i)
sac_data = np.load('model/sac_model/n_%d/sac_returns.pkl.npy' % i)
ppo_data = np.load('model/ppo_model/n_%d/ppo_returns.pkl.npy' % i)
qmix_data = np.load('model/qmix_model/n_%d/qmix_returns.pkl.npy' % i)

# Set font style
rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 12})

# Plotting
plt.figure(1)

# Only take the first 200 episodes
episodes = 200
maddpg_data = maddpg_data[:episodes]
sac_data = sac_data[:episodes]
ppo_data = ppo_data[:episodes]
qmix_data = qmix_data[:episodes]

# Plot data with improved style
plt.plot(maddpg_data, color='red', label='MARL', linestyle='-', alpha=0.7)
plt.plot(sac_data, color='blue', label='SAC', linestyle='-', alpha=0.7)
plt.plot(ppo_data, color='green', label='PPO', linestyle='-', alpha=0.7)
plt.plot(qmix_data, color='orange', label='QMIX', linestyle='-', alpha=0.7)

# Add title, legend, and labels
plt.title('Comparison of Returns')
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.legend(loc='lower left', fontsize='medium')

# Adjust axis limits and grid
#plt.xlim(0, len(maddpg_data))  # Adjust based on your data range
plt.grid(True, linestyle='--', alpha=0.5)

# Show plot
plt.tight_layout()  # Ensures all elements fit nicely in the plot area
plt.show()
