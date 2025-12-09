import numpy as np
import itertools, random
from monte_carlo import MonteCarlo


class MC_epsilon_greedy(MonteCarlo):
    def __init__(
            self,
            states,
            actions,
            epsilon = 0.5, 
            init_state_type = 'even',
            init_action_type = 'even',
            discount_rate = 1,
            alpha= 0.05
            ):
        super().__init__(
            states,
            actions,
            init_state_type,
            init_action_type,
            discount_rate,
            alpha
            )
        self.epsilon = epsilon
        
    def init_action_state(self,init_state_type , init_action_type , states, actions):
        # init state action value
        all_state_action_pairs = list(itertools.product(states, actions))
        if init_state_type == 'even':
            state_vals = [0.5] * len(all_state_action_pairs)
        else:
            state_vals = np.random.rand(len(all_state_action_pairs))
            state_vals /= state_vals.sum()

        self.val_states = {(state, action): float(state_val) for (state, action), state_val in zip(all_state_action_pairs, state_vals)}

        action_prob = 1/len(actions) 
        self.action_prob = {act: action_prob for act in actions}

    def epsilon_greedy(self, state, actions):
        rand = random.random()
        if rand > self.epsilon:
            return self.get_best_action(state, actions)
        return self.random_policy(actions)
    
    def get_best_action(self, state, actions):
        pos_val = [self.state_vals[(state, action)] for action in actions]
        return  actions[np.random.choice(np.flatnonzero(pos_val == pos_val.max()))]
        ## originally I was using dictionary to get all the relatable values than using max to get the best action )
        #   self.candidate = {action: self.q_value[(state , action)] for action in self.actions}
        #   max_key = max( self.candidate, key= self.candidate.get)
        #   return max_key
        
    def update(self, episodes):
        G_t = 0
        for states_id, action, reward in reversed(episodes):
            G_t = reward + self.discount_rate * G_t
            if (states_id,action) in self.val_states.keys():
                self.val_states[(states_id , action)] += self.alpha * ( G_t -  self.val_states[(states_id , action)])

 
