import numpy as np
import matplotlib.pyplot as plt
from Agent import Agent
from Env import Env
from LearningMethod import LearningMethod
from Policy import Policy
from Sim import Sim
from param_dic import ENV, AGENT, SIM
rng = np.random.default_rng()

env_dic = ENV['suboptimal']
sim_dic =  SIM
sim_dic['sim_size'] = 1000
avg_rewards = [0 for _ in range(8)]
aleph_g = 8

#main部
#RS
agt_dic = AGENT['RS_GRC']
agt_dic['policy_dic']['aleph_g'] = aleph_g
sim = Sim(env_dic = env_dic, agt_dic = agt_dic, sim_dic = sim_dic)
avg_rewards[0] = sim.exe_muti_sims()
agt_dic['policy_dic']['zeta'] = 10
sim = Sim(env_dic = env_dic, agt_dic = agt_dic, sim_dic = sim_dic)
avg_rewards[1] = sim.exe_muti_sims()

#RS(λ)
agt_dic = AGENT['RS_GRC_lamda']
agt_dic['policy_dic']['aleph_g'] = aleph_g
sim = Sim(env_dic = env_dic, agt_dic = agt_dic, sim_dic = sim_dic)
avg_rewards[2] = sim.exe_muti_sims()
agt_dic['policy_dic']['zeta'] = 10
sim = Sim(env_dic = env_dic, agt_dic = agt_dic, sim_dic = sim_dic)
avg_rewards[3] = sim.exe_muti_sims()

#RS GS
agt_dic = AGENT['RS_GRC']
agt_dic['policy_dic']['aleph_g'] = aleph_g
agt_dic['policy_dic']['sampling'] = 'off_policy'
sim = Sim(env_dic = env_dic, agt_dic = agt_dic, sim_dic = sim_dic)
avg_rewards[4] = sim.exe_muti_sims()
agt_dic['policy_dic']['zeta'] = 10
sim = Sim(env_dic = env_dic, agt_dic = agt_dic, sim_dic = sim_dic)
avg_rewards[5] = sim.exe_muti_sims()

#RS(λ) GS
agt_dic = AGENT['RS_GRC_lamda']
agt_dic['policy_dic']['aleph_g'] = aleph_g
agt_dic['policy_dic']['sampling'] = 'off_policy'
sim = Sim(env_dic = env_dic, agt_dic = agt_dic, sim_dic = sim_dic)
avg_rewards[6] = sim.exe_muti_sims()
agt_dic['policy_dic']['zeta'] = 10
sim = Sim(env_dic = env_dic, agt_dic = agt_dic, sim_dic = sim_dic)
avg_rewards[7] = sim.exe_muti_sims()

np.save('/Users/koshikawashunpei/Documents/高橋研/JSAI/プロットコード/plot_data/feeding_plot_zeta.npy', avg_rewards)

#plot部
fig, ax = plt.subplots(figsize = [12, 8])

label = ['RS ζ: 1', 'RS ζ: 10', 'RS(λ) ζ: 1', 'RS(λ) ζ: 10', 'RS GS ζ: 1', 'RS GS ζ: 10', 'RS(λ) GS ζ: 1', 'RS(λ) GS ζ: 10']
for i in range(8):
    ax.plot(avg_rewards[i], label = label[i])

ax.set_title('Feeding ground task', fontsize = 20)
ax.set_xlabel('Episode', fontsize = 20)
ax.set_ylabel('Rewards', fontsize = 20)
ax.legend(loc = 'lower right', fontsize = 15)

plt.show()
