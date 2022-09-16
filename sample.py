import numpy as np
import matplotlib.pyplot as plt
from Agent import Agent
from Env import Env
from LearningMethod import LearningMethod
from Policy import Policy
from Sim import Sim
from param_dic import ENV, AGENT, SIM
rng = np.random.default_rng()

agt_dic = AGENT['RS_GRC_lamda']
agt_dic['policy_dic']['aleph_g'] = 7.5
agt_dic['policy_dic']['gs_interval'] = 100

env_dic = ENV['maze']
sim_dic =  SIM
sim_dic['sim_size'] = 1000

#main部
avg_rewards = [0]
sim = Sim(env_dic = env_dic, agt_dic = agt_dic, sim_dic = sim_dic)
avg_rewards[0] = sim.exe_muti_sims()
#avg_rewards.append(sim.exe_muti_sims())

fig, ax = plt.subplots(figsize = [12, 8])

label = ['ε-greedy', 'RS', 'RS(λ)', 'RS GS-interval: 1', 'RS GS-interval: 10', 'RS(λ) GS-interval: 1', 'RS(λ) GS-interval: 10', 'RS ζ: 1', 'RS ζ: 10', 'RS(λ) ζ: 1', 'RS(λ) ζ: 10', 'RS GS ζ: 1', 'RS GS ζ: 10', 'RS(λ) GS ζ: 1', 'RS(λ) GS ζ: 10', 'RS(λ) GS interval: 50', 'RS(λ) GS interval: 5']

ax.plot(avg_rewards[0], label = 'RS(λ) GS interval: 100')

ax.set_title('Suboptimal World', fontsize = 20)
#ax.set_title('CliffWalk', fontsize = 20)
#ax.set_title('Maze task', fontsize = 20)
#ax.set_ybound(7.9, 8.01)
#ax.set_title('Feeding ground task', fontsize = 20)
ax.set_xlabel('Episode', fontsize = 20)
ax.set_ylabel('Rewards', fontsize = 20)
ax.legend(loc = 'lower right', fontsize = 15)

plt.show()
