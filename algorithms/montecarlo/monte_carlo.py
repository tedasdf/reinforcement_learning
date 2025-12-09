import numpy as np


class MonteCarlo():
    def __init__(
            self,
            states,
            actions,
            init_state_type = 'even',
            init_action_type = 'even',
            discount_rate = 1,
            alpha= 0.05
            ):
        self.discount_rate = discount_rate
        self.alpha = alpha

        self.init_action_state(init_state_type, init_action_type, states, actions)
        

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

    
    def random_policy(self, actions):
        return np.random.choice(actions, p=self.action_prob)


    def update(self, episodes):
        G_t = 0 
        for state_id , reward in reversed(episodes):
            G_t = reward + self.discount_rate * G_t

            if state_id in self.val_states.keys():
                self.val_states[state_id] += self.alpha * (G_t - self.val_states[state_id])
        