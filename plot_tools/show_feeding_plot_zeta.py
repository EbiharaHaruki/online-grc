import numpy as np
import matplotlib.pyplot as plt

avg_rewards = np.load('/Users/koshikawashunpei/Documents/高橋研/JSAI/プロットコード/plot_data/feeding_plot_zeta.npy')

#plot部
fig, ax = plt.subplots(figsize = [12, 8])

label = ['RS ζ: 1', 'RS ζ: 10', 'RS(λ) ζ: 1', 'RS(λ) ζ: 10', 'RS GS ζ: 1', 'RS GS ζ: 10', 'RS(λ) GS ζ: 1', 'RS(λ) GS ζ: 10']
for i in range(8):
    ax.plot(avg_rewards[i], label = label[i])

ax.tick_params(axis = 'x', labelsize = 25)
ax.tick_params(axis = 'y', labelsize = 25)
ax.set_title('Feeding ground task', fontsize = 30)
ax.set_xlabel('Episode', fontsize = 30)
ax.set_ylabel('Rewards', fontsize = 30)
ax.legend(ncol = 2, loc = 'lower right', fontsize = 25)

plt.show()
