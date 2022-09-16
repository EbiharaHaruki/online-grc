import numpy as np
import matplotlib.pyplot as plt

avg_rewards = np.load('/Users/koshikawashunpei/Documents/高橋研/JSAI/プロットコード/plot_data/feeding_plot_algorithm.npy')

#plot部zZ
fig, ax = plt.subplots(figsize = [12, 8])

label = ['ε-greedy', 'RS', 'RS GS interval: 1', 'RS(λ)', 'RS(λ) GS interval: 1']
for i in range(5):
    ax.plot(avg_rewards[i], label = label[i])

ax.tick_params(axis = 'x', labelsize = 25)
ax.tick_params(axis = 'y', labelsize = 25)
ax.set_title('Feeding ground task', fontsize = 30)
ax.set_xlabel('Episode', fontsize = 30)
ax.set_ylabel('Rewards', fontsize = 30)
ax.legend(loc = 'lower right', fontsize = 30)

plt.show()
