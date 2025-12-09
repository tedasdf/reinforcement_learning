

class OneDimensionEnv():
    def __init__(self):
        self.state_types = ['no' ,-2 , -1, 0, 1, 2, 'set']
        self.actions = {
            'down': -1,
            'up': 1
        }
        self.states_idx = 3
        
        # for MC to evaluate
        self.episodes = []

    def reset(self):
        self.episodes = []
        self.states_idx = 3
        return False, list(self.actions.keys()), self.state_types
    
    def step(self, act):
        current_state_value = self.state_types[self.states_idx]
        self.states_idx += act
        reward = 0
        terminal = False
        if self.states_idx == len(self.state_types)-1 or self.states_idx == 0:
            terminal = True
            if self.states_idx == len(self.state_types)-1:
                reward = 1
        next_state = self.state_types[self.states_idx]

        self.episodes.append((current_state_value, act, reward))
        return next_state, reward , terminal , list(self.actions.keys())