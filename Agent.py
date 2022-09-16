import numpy as np
from LearningMethod import LearningMethod
from Policy import Policy
rng = np.random.default_rng()

# 今回はtabulerしか扱わないのでQ値の配列はAgentに持たせることにする
class Agent():
    def __init__(self, width, height, start_pos, param_dic):
        self.width = width
        self.height = height
        self.start_pos = start_pos
        self.agt_pos = start_pos
        self.num_action = param_dic['num_action']
        self.Q = np.zeros([width, height, self.num_action])
        self.param_dic = param_dic

        self.learning_method = LearningMethod(param_dic['lm_dic'], width, height, self.num_action)
        self.policy = Policy(param_dic['policy_dic'], self.num_action, width, height)

    def initialize(self):
        self.Q = np.zeros([self.width, self.height, self.num_action])
        self.agt_pos = self.start_pos
        self.policy = Policy(self.param_dic['policy_dic'], self.num_action, self.width, self.height)

    def make_action(self):
        a = self.policy.policy(self.agt_pos, self.Q)
        return a

    def make_sample(self):
        a = self.policy.sampling_policy(self.agt_pos, self.Q)
        return a

    def gs_update(self, a, r, s_next):
        sx = self.agt_pos[0]; sy = self.agt_pos[1];
        self.agt_pos = s_next

    def update(self, a, r, s_next):
        sx = self.agt_pos[0]; sy = self.agt_pos[1];
        if self.param_dic['policy_dic']['policy'] == 'RS_GRC':
            self.policy.update_tau(self.Q, sx, sy, a, s_next)

        #self.Q[sx, sy, a] = self.learning_method.learning_method(self.Q, self.agt_pos, a, r, s_next)
        self.Q = self.learning_method.learning_method(self.Q, self.agt_pos, a, r, s_next)
        self.agt_pos = s_next
