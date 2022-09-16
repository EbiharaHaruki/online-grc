import numpy as np
import matplotlib.pyplot as plt

#収束しきってそうなので500で切ってる
avg_rewards = np.load('/Users/koshikawashunpei/Documents/高橋研/JSAI/プロットコード/plot_data/maze_plot_algorithm.npy')[:,:500]

#plot部zZ
fig, ax = plt.subplots(figsize = [12, 8])

#全体のプロット
"""
label = ['ε-greedy', 'RS', 'RS GS interval: 1', 'RS(λ)', 'RS(λ) GS interval: 1']
for i in range(5):
    ax.plot(avg_rewards[i], label = label[i])
"""

#RS vs RS GSのプロット
"""
ax.plot(avg_rewards[1], label = 'RS', color = 'tab:orange')
ax.plot(avg_rewards[2], label = 'RS GS interval: 1', color = 'tab:green')
"""

#RS(λ) vs RS(λ) GSのプロット

ax.plot(avg_rewards[3], label = 'RS(λ)', color = 'tab:red')
ax.plot(avg_rewards[4], label = 'RS(λ) GS interval: 1', color = 'tab:purple')


ax.tick_params(axis = 'x', labelsize = 25)
ax.tick_params(axis = 'y', labelsize = 25)
ax.set_title('Maze task', fontsize = 30)
ax.set_xlabel('Episode', fontsize = 30)
ax.set_ylabel('Rewards', fontsize = 30)
ax.legend(loc = 'lower right', fontsize = 30)

plt.show()
