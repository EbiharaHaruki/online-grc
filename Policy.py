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


            #割引なしtau
            self.tau = np.zeros([width, height, num_action])
            #割引ありtau
            self.alpha_tau = 0.1
            self.gamma_tau = 0.9
            self.tau_current = np.zeros([width, height, num_action])
            self.tau_post = np.zeros([width, height, num_action])

            self.zeta = param_dic['zeta']   #本来zetaは配列であるが本実験では全て同じ値であるのでintで保持する。
            self.aleph_g = param_dic['aleph_g']
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
            a = np.random.choice(4)
        else:
            a = np.random.choice(np.where(Q[sx, sy] == max(Q[sx, sy]))[0])
        return a

    def softmax(self, s, Q):
        pass

    def rs_grc(self, s, Q):
        sx = s[0];  sy = s[1];

        d_g = min(self.e_g - self.aleph_g, 0)
        aleph = max(Q[sx, sy]) - self.zeta*d_g

        self.rs[sx, sy] = self.tau[sx, sy]*(Q[sx, sy] - aleph)
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
        self.tau_post[sx, sy, a] += self.alpha_tau*(self.gamma_tau*self.tau[s_next[0], s_next[1], a_update] - self.tau_post[sx, sy, a])
        self.tau[sx, sy, a] = self.tau_current[sx, sy, a] + self.tau_post[sx, sy, a]

    def e_greedy_update(self, *args):
        pass
