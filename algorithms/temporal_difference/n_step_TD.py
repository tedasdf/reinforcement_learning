import numpy as np


class n_step_TD():
    def __init__(
            self,
            states,
            actions,
            n,
            init_state_type = 'even',
            init_action_type = 'even',
            discount_rate = 1,
            alpha= 0.05
            ):
        self.discount_rate = discount_rate
        self.alpha = alpha

        self.n = n
        
        self.init_action_state(init_state_type, init_action_type, states, actions)
        
    def random_policy(self, actions):
        return np.random.choice(actions, p=self.action_prob)

    def init_action_state(self, init_state_type , init_action_type , states, actions):
        # init states value 
        if init_state_type == 'even':
            state_val = 0.5
            self.val_states = {state: state_val for state in states} 
        else:
            # random init 
            state_vals = np.random.rand(len(states))
            state_vals /= state_vals.sum()

            self.val_states = {state: float(val) for state , val in zip(states,state_vals)}
        

        # # init action probabilities 
        # if init_action_type == 'even':
        action_prob = 1/len(actions) 
        self.action_prob = {act: action_prob for act in actions}

    
    def policy(self, actions ):
        return np.random.choice(actions, p=self.action_prob)


    def update(self):
        T = len(self.episodes)
        t = T - self.n

        if t < 0:
            return

        S_T = self.episodes[-1][0] 
        G_t = self.val_states.get(S_T, 0) # G_t starts as V(S_T)
        
        for i in range(T - 1, t - 1, -1):
            reward = self.episodes[i][1] # Reward is at position [1]
            G_t = reward + self.discount_rate * G_t
            
        S_update = self.episodes[t][0]
        
        V_old = self.val_states.get(S_update, 0)
        self.val_states[S_update] += self.alpha * (G_t - V_old)

    def step_update(self, actions , state_last, rew_last):
        selected_action = self.policy(actions)
        self.episodes.append((state_last , rew_last))

        if len(self.episodes) >= self.n:
            self.update()
        return selected_action