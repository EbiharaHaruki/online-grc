import numpy as np
from Agent import Agent
from Env import Env
from tqdm import tqdm
from Policy import Policy

rng = np.random.default_rng()

class Sim():
    def __init__(self, env_dic, agt_dic, sim_dic):
        #self.sin_size = sim_dic['sim_size']
        self.start_pos = env_dic['start_pos']
        self.epi_size = sim_dic['epi_size']
        self.step_size = sim_dic['step_size']
        self.sim_size = sim_dic['sim_size']
        self.agt = Agent(width = env_dic['width'], height = env_dic['height'], start_pos = self.start_pos, param_dic = agt_dic)
        self.env = Env(env_dic)

        self.avg_rewards = np.zeros(self.epi_size)
        self.avg_rewards_greedy = np.zeros(self.epi_size)
        self.rewards_ex = np.zeros(self.epi_size)
        self.avg_rewards_ex = np.zeros(self.epi_size)
        self.alephs = np.zeros(self.epi_size)
        self.avg_alephs = np.zeros(self.epi_size)

    def exe_sim(self):
        r = 0
        gs_count = 0
        rewards = []
        rewards_greedy = []
        var = 0
        for n_epi in range(self.epi_size):
            is_terminal = False
            reward = 0
            reward_greedy = 0
            #a_next = self.agt.make_action_Q(self.start_pos)#
            for n_step in range(self.step_size):
                if is_terminal:
                    #self.agt.printQ()
                    #print(r)
                    self.agt.agt_pos = self.start_pos
                    self.env.agt_pos = self.start_pos
                    #print(f"reward = {reward}")
                    break
                a = self.agt.make_action(n_epi,r,var)
                #a = a_next#
                s_next, r, is_terminal = self.env.update_env(self.agt.agt_pos, a)
                reward += r
                #a_next = self.agt.make_action_Q(s_next)#
                self.agt.update(a, r, s_next)
            
            if n_epi % 10 == 9:
                is_terminal = False
                for n_step in range(self.step_size):
                    if is_terminal:
                        self.agt.agt_pos = self.start_pos
                        self.env.agt_pos = self.start_pos
                        #print(f"reward_greedy = {reward_greedy}")
                        break
                    a = self.agt.make_action_greedy()
                    s_next, r, is_terminal = self.env.update_env(self.agt.agt_pos, a)
                    #print(f"r = {r}")
                    reward_greedy += r
                    self.agt.update(a, r, s_next)
            else:
                    r = 0
                    reward_greedy += r
                
            #greedy_sampling
            if self.agt.policy.sampling == 'off_policy':
                gs_count += 1
                if self.agt.policy.gs_interval <= gs_count:
                    is_terminal = False
                    gs_count = 0
                    reward_gs = 0
                    for n_step in range(self.step_size):
                        if is_terminal:
                            self.agt.agt_pos = self.start_pos
                            self.env.agt_pos = self.start_pos
                            break
                        a = self.agt.make_sample()
                        s_next, r, is_terminal = self.env.update_env(self.agt.agt_pos, a)
                        reward_gs += r
                        self.agt.gs_update(a, r, s_next)
                    self.agt.policy.update(reward_gs)

            else:
                self.agt.policy.update(reward)
            
            if n_epi % 10 == 0:
                self.rewards_ex[n_epi] = reward
            else:
                self.rewards_ex[n_epi] = 0
            #print(f"reward_greedy = {reward_greedy},reward = {reward}")
            rewards_greedy.append(reward_greedy)
            rewards.append(reward)
            if n_epi >= 10:
                i = rewards[-10:]
            else:
                i = rewards
            var = np.var(i)
            #print(i,var)
            #aleph = self.agt.policy.aleph_g
            #self.alephs[n_epi] = aleph
        #print(f"rewards_greedy = {rewards_greedy}")
        #var = np.var(rewards)
        #print(f"var = {var}")
        return rewards,rewards_greedy

    def exe_muti_sims(self):
        for n_sim in tqdm(range(self.sim_size)):
            self.agt.initialize()
            self.env.initialize()
            rewards,rewards_greedy = self.exe_sim()  #ここで1回sim回ってる
            self.avg_rewards += (rewards - self.avg_rewards)/(n_sim+1)
            self.avg_rewards_greedy += (rewards_greedy - self.avg_rewards_greedy)/(n_sim+1)
            #self.avg_alephs += (self.alephs - self.avg_alephs)/(n_sim+1)
            #self.avg_rewards_ex += (self.rewards_ex - self.avg_rewards_ex)/(n_sim+1)

        return self.avg_rewards,self.avg_rewards_greedy,self.avg_alephs
