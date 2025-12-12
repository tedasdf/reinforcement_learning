

class OneDimensionEnv:
    def __init__(self):
        self.state_types = ['no', -2, -1, 0, 1, 2, 'set']
        self.actions = {
            'down': -1,
            'up': 1
        }
        self.states_idx = 3

    @property
    def action_space(self):
        return list(self.actions.keys())
  
    def reset(self):
        self.states_idx = 3
        return self.state_types[self.states_idx]

    def step(self, action):
        current_state_value = self.state_types[self.states_idx]
        self.states_idx += self.actions[action]  # map action string to change

        # Clip index
        self.states_idx = max(0, min(self.states_idx, len(self.state_types) - 1))

        reward = 0
        done = False
        if self.states_idx == 0 or self.states_idx == len(self.state_types) - 1:
            done = True
            if self.states_idx == len(self.state_types) - 1:
                reward = 1

        next_state = self.state_types[self.states_idx]

        return next_state, reward, done, {}
        