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
#agt_dic = AGENT['e_greedy']
agt_dic['policy_dic']['aleph_g'] = 0
agt_dic['policy_dic']['gs_interval'] = 1

env_dic = ENV['suboptimal']
sim_dic =  SIM
sim_dic['sim_size'] = 1000

#main部
avg_rewards = [0]
avg_rewards_greedy = [0]
avg_alephs = [0]
avg_rewards_ex = [0]
xdata = []
aleph_opt = [0]
aleph_opt_greedy = []
aleph_opt_greedy_01 = [0]
aleph_dl_greedy = []
aleph_dl_greedy_01 = [0]
aleph_dl = [0]
aleph_high = [0]
q_learning = [0]
q_learning_greedy = []
q_learning_greedy_01 = [0]
q_learning_annealing = [0]

sim = Sim(env_dic = env_dic, agt_dic = agt_dic, sim_dic = sim_dic)
#avg_rewards[0],avg_rewards_greedy[0],avg_alephs[0] = sim.exe_muti_sims()
#np.save('aleph_dl',avg_rewards[0])
#np.save('aleph_dl_greedy',avg_rewards_greedy[0])

#aleph_high[0] = np.load('aleph_high.npy')
aleph_opt[0] = np.load('aleph_opt_maze.npy')
aleph_opt_greedy_01[0] = np.load('aleph_opt_greedy_maze.npy')
aleph_dl[0] = np.load('aleph_dl.npy')
aleph_dl_greedy_01[0] = np.load('aleph_dl_greedy_maze.npy')
q_learning[0] = np.load('q_learning_0.1.npy')
q_learning_greedy_01[0] = np.load('q_learning_greedy_maze.npy')
q_learning_annealing[0] = np.load('q_learning_annealing_maze.npy')



for i in range(500):
    if i == 0:
        xdata.append(i)
        aleph_dl_greedy.append(aleph_dl_greedy_01[0][i])
        q_learning_greedy.append(q_learning_greedy_01[0][i])
        aleph_opt_greedy.append(aleph_opt_greedy_01[0][i])
    elif i % 10 == 9:
        xdata.append(i)
        aleph_dl_greedy.append(aleph_dl_greedy_01[0][i])
        q_learning_greedy.append(q_learning_greedy_01[0][i])
        aleph_opt_greedy.append(aleph_opt_greedy_01[0][i])
        

#print(aleph_dl)
#print(aleph_opt)
#print(avg_rewards_ex)
#print(f"aleph_dl = {aleph_dl[0][0]}")

#print(f"aleph_dl_greedy_01 = {aleph_dl_greedy_01[0]},dl_greedy = {aleph_dl_greedy}")
#print(f'avg_alephs = {avg_alephs}')
#avg_rewards.append(sim.exe_muti_sims())

fig, ax = plt.subplots(figsize = [12, 8])

label = ['ε-greedy', 'RS', 'RS(λ)', 'RS GS-interval: 1', 'RS GS-interval: 10', 'RS(λ) GS-interval: 1', 'RS(λ) GS-interval: 10', 'RS ζ: 1', 'RS ζ: 10', 'RS(λ) ζ: 1', 'RS(λ) ζ: 10', 'RS GS ζ: 1', 'RS GS ζ: 10', 'RS(λ) GS ζ: 1', 'RS(λ) GS ζ: 10', 'RS(λ) GS interval: 50', 'RS(λ) GS interval: 5']

#ax.plot(aleph_dl[0], label = 'Online-GRC')
#ax.plot(avg_alephs[0], label = 'aleph_g')
#ax.plot(aleph_opt[0], label = 'GRC')
#ax.plot(q_learning_annealing[0], label = 'Qlearning')

ax.plot(xdata,aleph_dl_greedy, label = 'Online-GRC_greedy')
ax.plot(xdata,aleph_opt_greedy, label = 'GRC_greedy')
ax.plot(xdata,q_learning_greedy, label = 'Qlearning_greedy')



#ax.set_title('maze', fontsize = 20)
#ax.set_title('CliffWalk', fontsize = 20)
#ax.set_title('Maze task', fontsize = 20)
#ax.set_ybound(7.9, 8.01)
#ax.set_title('Feeding ground task', fontsize = 20)
ax.set_xlabel('Episodes', fontsize = 20)
ax.set_ylabel('Returns', fontsize = 20)
ax.legend(loc = 'lower right', fontsize = 15,frameon=False)

plt.show()
