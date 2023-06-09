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
        self.alephs = np.zeros(self.epi_size)
        self.avg_alephs = np.zeros(self.epi_size)

    def exe_sim(self):
        r = 0
        rewards = np.zeros(self.epi_size)
        gs_count = 0
        for n_epi in range(self.epi_size):
            is_terminal = False
            reward = 0
            for n_step in range(self.step_size):
                if is_terminal:
                    #print(f'self.agt.agt_pos = {self.agt.agt_pos}')
                    #aleph1 = self.agt.return_aleph()
                    #print(f'aleph1 = {aleph1}')
                    #self.agt.print_Q()
                    #self.agt.print_maxQ()
                    self.agt.agt_pos = self.start_pos
                    self.env.agt_pos = self.start_pos
                    break
                a = self.agt.make_action(n_epi,self.step_size,r)#ここでαなどの計算をしている
                s_next, r, is_terminal = self.env.update_env(self.agt.agt_pos, a)
                #self.agt.print_maxQ()
                #if r != 0:
                    #self.agt.print_Q()
                    #self.agt.print_maxQ()
                    #print(f'r = {r}')
                reward += r
                self.agt.update(a, r, s_next)
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

            rewards[n_epi] = reward
            aleph = self.agt.policy.aleph_g
            #print(f'aleph = {aleph}')
            #print(f'self.agt.agt_pos = {self.agt.agt_pos}')
            self.alephs[n_epi] = aleph
        return rewards

    def exe_muti_sims(self):
        for n_sim in tqdm(range(self.sim_size)):
            self.agt.initialize()
            self.env.initialize()
            rewards = self.exe_sim()  #ここで1回sim回ってる
            self.avg_rewards += (rewards - self.avg_rewards)/(n_sim+1)
            self.avg_alephs += (self.alephs - self.avg_alephs)/(n_sim+1)

        return self.avg_rewards,self.avg_alephs
