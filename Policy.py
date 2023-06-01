import numpy as np
rng = np.random.default_rng()


class Policy():
    def __init__(self, param_dic, num_action, width, height):
        self.sampling = 'on_policy'
        self.set_policy(param_dic, num_action, width, height)

    def set_policy(self, param_dic, num_action, width, height):
        self.n_g = 0
        if param_dic['policy'] == 'greedy':
            self.policy = self.greedy
            self.update = self.greedy_update

        elif param_dic['policy'] == 'e_greedy':
            self.policy = self.e_greedy
            self.update = self.e_greedy_update
            self.epsilon = param_dic['epsilon']

        elif param_dic['policy'] == 'softmax':
            self.policy = self.softmax
            self.update = self.softmax_update

        elif param_dic['policy'] == 'RS_GRC':
            self.policy = self.rs_grc
            self.update = self.rs_grc_update
            self.rs = np.zeros([width, height, num_action])
            self.sampling = param_dic['sampling']
            self.sampling_policy = self.greedy
            self.gs_interval = param_dic['gs_interval']
            self.gamma_g = 0.9
            self.n_tmp = 10
            self.t_tmp = 0
            self.n_pre = 0
            self.alpha = 1
            self.n_over = 1
            self.n_max_r = 0
            self.max_r = -10000
            self.reward = 0
            self.reward_n = 0
            self.R = 0


            #割引なしtau
            self.tau = np.zeros([width, height, num_action])
            #割引ありtau
            self.alpha_tau = 0.1
            self.gamma_tau = 0.9
            self.tau_current = np.zeros([width, height, num_action])
            self.tau_post = np.zeros([width, height, num_action])

            self.zeta = param_dic['zeta']   #本来zetaは配列であるが本実験では全て同じ値であるのでintで保持する。
            self.aleph = 0
            self.aleph_g = param_dic['aleph_g'] #aleph_g = 7.5(opt)
            self.gamma_g = param_dic['gamma_g']
            self.e_g = 0
            self.e_tmp = 0


    def greedy(self, s, Q):
        sx = s[0];  sy = s[1];
        a = np.random.choice(np.where(Q[sx, sy] == max(Q[sx, sy]))[0])
        return a

    def e_greedy(self, s, Q):
        sx = s[0];  sy = s[1];
        if self.epsilon <= rng.random():
            self.epsilon = self.epsilon * 1.001
            #print(self.epsilon)
            a = np.random.choice(4)
        else:
            a = np.random.choice(np.where(Q[sx, sy] == max(Q[sx, sy]))[0])
        return a
    
        
    def softmax(self, s, Q):
        pass

    def rs_grc(self, s,Q,n_epi,reward,var):#s=agt_pos
        sx = s[0];  sy = s[1];
        #print(f'self.max_r = {self.max_r}')
        #print(f'n = {n},n_pre = {self.n_pre}')
        #ここの計算を行った後、報酬を貰い、情報がリセットされてしまうため
        #print(f"epi = {n_epi},epi_pre = {self.n_pre}")
        
        self.reward += reward
            #print(f'self.R = {self.R}')
        #ステップ更新ではなくエピソード更新を行うため
        if n_epi != self.n_pre:
            if self.max_r < self.reward:
                self.max_r = self.reward
                #print(f"max_r = {self.max_r},n_epi = {n_epi}")
            #print(f'reward = {self.reward},aleph_g = {self.aleph_g}')
            self.R = ((n_epi-1)/n_epi)*self.R + (1/n_epi)*self.reward
            #print(f'R = {self.R}')
            #if self.n_max_r < 200:#最大の報酬の回数
            #print(f"self.e_g = {self.e_g},reward={self.reward}")
            if self.aleph_g > self.reward:
                if self.aleph_g > self.max_r:
                    self.aleph_g = self.max_r + 0.9 * (abs(self.aleph_g) - abs(self.reward))
                    #self.aleph_g = self.aleph_g - 1/50 * abs(self.max_r)
                else:
                    self.aleph_g = self.max_r
                #print(f"dic  aleph_g = {self.aleph_g},n_epi = {n_epi},r = {self.reward},max_r = {self.max_r}")
                #print(f'if aleph_g = {self.aleph_g},reward = {self.reward}')
            #ℵ_Gより報酬の方が大きかったら希求水準を引き上げる
            else:
                #print(f'aleph_g = {self.aleph_g},e_g = {self.e_g}')
                #print(f'else aleph_g = {self.aleph_g},reward = {self.reward}')
                #print(f"var = {var},n_epi = {n_epi}")
                self.aleph_g = self.reward + abs(self.reward) * ((var ** 0.5) /self.n_over)
                #print(f"inc  aleph_g = {self.aleph_g},n_epi = {n_epi}")
                #self.aleph_g = self.reward + abs(self.reward) * var
                #print(f"////////////////////{self.aleph_g},/////////////////////")
                self.n_over += 1#上回った回数
                #print(f'self.n_over = {self.n_over}')
            self.reward = 0
            self.n_pre = n_epi
            #else:
                #self.aleph_g = self.max_r
                #self.n_pre = n_epi#policy側でのエピソードの値を更新
                #if self.max_r < self.reward:
                    #self.max_r = self.reward#一番大きい報酬の値を取得
                #if self.max_r == self.e_g:
                    #self.n_max_r += 1
                    #print(f'n_max_r = {self.n_max_r},self.max_r = {self.max_r}')
                #print(f'self.max_r = {self.max_r}')
                #print(f'self.e_g = {self.e_g}')
            
        #self.aleph_g = -12
        #print(f'aleph_g = {self.aleph_g}')
        d_g = min(self.e_g - self.aleph_g, 0)
        #print(f'maxQ = {max(Q[sx,sy])}')
        self.aleph = max(Q[sx, sy]) - self.zeta*d_g
        #print(f'self.aleph = {self.aleph}')

        self.rs[sx, sy] = self.tau[sx, sy]*(Q[sx, sy] - self.aleph)
        a = np.random.choice(np.where(self.rs[sx, sy] == max(self.rs[sx, sy]))[0])

        return a


    def rs_grc_update(self, reward):
        if self.sampling == 'off_policy':
            self.e_g = reward
        else:
            #1-step Eg
            self.e_g = reward

            #simple avg Eg
            #self.n_g += 1
            #self.e_g += (reward - self.e_g)/self.n_g

            #discount Eg
            #self.e_g = (reward + self.gamma_g*(self.n_g*self.e_g))/(1.0 + self.gamma_g*self.n_g)
            #self.n_g = 1.0 + self.gamma_g*self.n_g

            #discount and temporal Eg
            #if self.t_tmp > self.n_tmp:
            #    self.t_tmp = 0
            #    self.e_tmp = 0
            #self.t_tmp += 1
            #self.e_tmp += (reward - self.e_tmp)/self.t_tmp
            #self.e_g = (self.e_tmp + self.gamma_g*(self.n_g*self.e_g))/(1.0 + self.gamma_g*self.n_g)
            #self.n_g = 1.0 + self.gamma_g*self.n_g



    def update_tau(self, Q, sx, sy, a, s_next):
        #illegal update
        #self.tau[sx, sy] += 1

        #割引なしtau
        #self.tau[sx, sy, a] += 1

        #割引ありtau
        a_update = np.random.choice(np.where(Q[sx, sy] == max(Q[sx, sy]))[0])
        self.tau_current[sx, sy, a] += 1
        #self.tau_post[sx, sy, a] += self.alpha_tau*(self.gamma_tau*self.tau[s_next[0], s_next[1], a_update] - self.tau_post[sx, sy, a])
        self.tau_post[sx, sy, a] = 0
        self.tau[sx, sy, a] = self.tau_current[sx, sy, a] + self.tau_post[sx, sy, a]

    def e_greedy_update(self, *args):
        pass
