import numpy as np
import matplotlib.pyplot as plt
from Agent import Agent
from Env import Env
from LearningMethod import LearningMethod
from Policy import Policy
from Sim import Sim
from param_dic import ENV, AGENT, SIM
rng = np.random.default_rng()

env_dic = ENV['maze']
sim_dic = SIM
sim_dic['sim_size'] = 1000
sim_dic['epi_size'] = 500
avg_rewards = [0,0,0,0,0]
aleph_g = -12
#main部
agt_dic = AGENT['e_greedy']
sim = Sim(env_dic = env_dic, agt_dic = agt_dic, sim_dic = sim_dic)
avg_rewards[0] = sim.exe_muti_sims()

agt_dic = AGENT['RS_GRC']
agt_dic['policy_dic']['aleph_g'] = aleph_g
sim = Sim(env_dic = env_dic, agt_dic = agt_dic, sim_dic = sim_dic)
avg_rewards[1] = sim.exe_muti_sims()

agt_dic = AGENT['RS_GRC']
agt_dic['policy_dic']['aleph_g'] = aleph_g
agt_dic['policy_dic']['sampling'] = 'off_policy'
agt_dic['policy_dic']['gs_interval'] = 1
sim = Sim(env_dic = env_dic, agt_dic = agt_dic, sim_dic = sim_dic)
avg_rewards[2] = sim.exe_muti_sims()

agt_dic = AGENT['RS_GRC_lamda']
agt_dic['policy_dic']['aleph_g'] = aleph_g
sim = Sim(env_dic = env_dic, agt_dic = agt_dic, sim_dic = sim_dic)
avg_rewards[3] = sim.exe_muti_sims()

agt_dic = AGENT['RS_GRC_lamda']
agt_dic['policy_dic']['aleph_g'] = aleph_g
agt_dic['policy_dic']['sampling'] = 'off_policy'
agt_dic['policy_dic']['gs_interval'] = 1
sim = Sim(env_dic = env_dic, agt_dic = agt_dic, sim_dic = sim_dic)
avg_rewards[4] = sim.exe_muti_sims()

np.save('/Users/koshikawashunpei/Documents/高橋研/JSAI/プロットコード/plot_data/maze_plot_algorithm.npy', avg_rewards)

#plot部
fig, ax = plt.subplots(figsize = [12, 8])

label = ['ε-greedy', 'RS', 'RS GS-interval: 1', 'RS(λ)', 'RS(λ) GS-interval: 1']
for i in range(5):
    ax.plot(avg_rewards[i], label = label[i])

ax.set_title('Maze task', fontsize = 20)
ax.set_xlabel('Episode', fontsize = 20)
ax.set_ylabel('Rewards', fontsize = 20)
ax.legend(loc = 'lower right', fontsize = 15)

plt.show()
