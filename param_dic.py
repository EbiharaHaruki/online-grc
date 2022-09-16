ENV = {
    'maze': {
        'task': 'waka_GridWorld',
        'width': 7,
        'height': 7,
        'start_pos': (0,0),
        'wall_pos': {(0,2), (1,2), (2,4), (2,5), (3,3), (4,5), (5,1), (5,2), (5,3), (5,4)},
        'reward_dic': {(x, y): -1 for x in range(-1, 8) for y in range(-1, 8)},
        'terminal_pos': {(6,6)}
    },
    'suboptimal': {
        'task': 'suboptimal',
        'width': 9,
        'height': 9,
        'start_pos': (4,4),
        'wall_pos': set(),
        'reward_dic': {(0, 2): 3, (0, 6): 6, (2, 0): 8, (2, 8): 2, (6, 0): 1, (6, 8): 7, (8, 2): 5, (8, 6): 4},
        'terminal_pos': {(0, 2), (0, 6), (2, 0), (2, 8), (6, 0), (6, 8), (8, 2), (8, 6)}
    }
}

AGENT = {
    'e_greedy': {
        'num_action': 4,
        'policy_dic': {'policy': 'e_greedy', 'epsilon': 0.1},
        'lm_dic': {'learning_method': 'Q', 'alpha': 0.1, 'gamma': 0.9}
    },
    'RS_GRC': {
        'num_action': 4,
        'policy_dic': {'policy': 'RS_GRC', 'zeta': 1, 'aleph_g': 7.5, 'gamma_g': 0.9, 'sampling': 'on_policy', 'gs_interval': 1},
        'lm_dic': {'learning_method': 'Q', 'alpha': 0.1, 'gamma': 0.9}
    },
    'RS_GRC_lamda': {
        'num_action': 4,
        'policy_dic': {'policy': 'RS_GRC', 'zeta': 1, 'aleph_g': 7.5, 'gamma_g': 0.9, 'sampling': 'on_policy', 'gs_interval': 1},
        'lm_dic': {'learning_method': 'q_lamda', 'alpha': 0.1, 'gamma': 0.9, 'lamda': 0.9}
    }
}

SIM = {
    'sim_size': 100,
    'step_size': 1000,
    'epi_size': 1000
}
