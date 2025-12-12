

class nStateEnv:
    def __init__(self, n=50):
        self.n = n
        self.state_types = list(range(n))

        self.actions = {
            'down': -1,
            'up': 1
        }
        self.states_idx = n//2

    @property
    def action_space(self):
        return list(self.actions.keys())
  
    def reset(self):
        self.states_idx = self.n//2
        return self.state_types[self.states_idx]

    def step(self, action):
        self.states_idx += self.actions[action]  # map action string to change

        # Clip index
        self.states_idx = max(0, min(self.states_idx, self.n - 1))

        done = self.states_idx == 0 or self.states_idx == self.n - 1
        reward = 1 if self.states_idx == self.n-1 else 0


        next_state = self.state_types[self.states_idx]

        return next_state, reward, done, {}
        