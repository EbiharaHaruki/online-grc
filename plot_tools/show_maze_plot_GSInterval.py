import numpy as np
import matplotlib.pyplot as plt

avg_rewards = np.load('/Users/koshikawashunpei/Documents/高橋研/JSAI/プロットコード/plot_data/maze_plot_GSInterval.npy')[:,:500]

#plot部
fig, ax = plt.subplots(figsize = [12, 8])

label = ['RS(λ) GS interval: 100', 'RS(λ) GS interval: 50', 'RS(λ) GS interval: 10', 'RS(λ) GS interval: 5', 'RS(λ) GS interval: 1']
for i in range(5):
    ax.plot(avg_rewards[i], label = label[i])

ax.tick_params(axis = 'x', labelsize = 25)
ax.tick_params(axis = 'y', labelsize = 25)
ax.set_title('Maze task', fontsize = 30)
ax.set_xlabel('Episode', fontsize = 30)
ax.set_ylabel('Rewards', fontsize = 30)
ax.legend(loc = 'lower right', fontsize = 25)
#ax.legend(loc = 'upper right', fontsize = 30)

plt.show()
