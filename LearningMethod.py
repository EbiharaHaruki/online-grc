import numpy as np

class LearningMethod():
    def __init__(self, param_dic, width, height, num_action):
        self.set_value_estimate_method(param_dic, width, height, num_action)

    def set_value_estimate_method(self, param_dic, width, height, num_action):
        self.width = width
        self.height = height
        if param_dic['learning_method'] == 'Q':
            self.learning_method = self.q_learning
            self.alpha = param_dic['alpha']
            self.gamma = param_dic['gamma']
        elif param_dic['learning_method'] == 'q_lamda':
            self.learning_method = self.q_lamda
            self.alpha = param_dic['alpha']
            self.gamma = param_dic['gamma']
            self.lamda = param_dic['lamda']
            self.e = np.zeros((width, height, num_action))


    def q_learning(self, Q, s, a, r, s_next):
        #update Q(methodとして外に出してもいいかもしれないがsarsaと微妙に違うのでいいかな)
        #q_sa = Q[s[0], s[1], a] + self.alpha*(r+self.gamma*max(Q[s_next[0], s_next[1]]) - Q[s[0], s[1], a])
        #return q_sa
        Q[s[0], s[1], a] += self.alpha*(r+self.gamma*max(Q[s_next[0], s_next[1]]) - Q[s[0], s[1], a])
        return Q

    def q_lamda(self, Q, s, a, r, s_next):
        #eの更新

        #これは多分Sarsaの方
        #self.e *= self.gamma*self.lamda
        #self.e[s[0], s[1], a] += 1

        #Q(λ)
        if a == np.random.choice(np.where(Q[s[0], s[1]] == max(Q[s[0], s[1]]))[0]):
            self.e *= self.gamma*self.lamda
            self.e[s[0], s[1], a] += 1
        else:
            self.e *= 0
            self.e[s[0], s[1], a] += 1

        td_error = r+self.gamma*max(Q[s_next[0], s_next[1]]) - Q[s[0], s[1], a]
        Q += self.alpha*td_error*self.e
        return Q
