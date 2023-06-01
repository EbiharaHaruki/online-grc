import numpy as np

#状態sの表現をどうするか
#左上が原点；
class Env():
    def __init__(self, env_dic = None):
        self.task = env_dic['task']
        self.width = env_dic['width']
        self.height = env_dic['height']
        self.start_pos = env_dic['start_pos']
        self.agt_pos = env_dic['start_pos']
        self.wall_pos = env_dic['wall_pos'] #wall_posのSet
        self.reward_dic = env_dic['reward_dic'] #{pos: 報酬値}
        self.reward_pos = set(self.reward_dic.keys())
        self.terminal_pos = env_dic['terminal_pos']

    def initialize(self):
        self.agt_pos = self.start_pos

    def get_reward(self, s_next): #呼び出す前にmoveableであるかのチェック必要
        if s_next in self.reward_pos:
            return self.reward_dic[s_next]
        return 0

    def is_movable(self, s_next):
        #境界判定
        next_x = s_next[0]; next_y = s_next[1];
        if next_x < 0 or next_x > self.width-1:
            return False
        elif next_y < 0 or next_y > self.height-1:
            return False

        if s_next in self.wall_pos:
            return False

        return True

    def update_env(self, s, a):
        sx = s[0];  sy = s[1];
        if a == 0:
            ax = 0;  ay = -1;#↑
        elif a == 1:
            ax = 1;  ay = 0;#→
        elif a == 2:
            ax = 0;  ay = 1;#↓
        elif a == 3:
            ax = -1;  ay = 0;#←

        s_next = (sx+ax, sy+ay)
        r = self.get_reward(s_next)
        if self.is_movable(s_next):
            is_terminal = s_next in self.terminal_pos
            self.agt_pos = s_next
        else:
            s_next = s
            is_terminal = False

        return s_next, r, is_terminal
