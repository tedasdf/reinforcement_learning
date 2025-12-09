
import itertools, random
import numpy as np
from temporal_difference import n_step_TD

class n_step_Sarsa(n_step_TD):
    def __init__(self, epsilon, n):
        self.epsilon = epsilon
        
        self.episodes = []
        self.n  = n

          
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
    
    def update(self):
        T = len(self.episodes)
        t = T - self.n

        if t < 0:
            return

        S_T, A_T , _ = self.episodes[-1] 
        
        G_t = self.val_states.get((S_T , A_T), 0) # G_t starts as V(S_T)
        
        for i in range(T - 1, t - 1, -1):
            reward = self.episodes[i][1] # Reward is at position [1]
            G_t = reward + self.discount_rate * G_t
            
        S_update = self.episodes[t][0]
        A_update = self.episodes[t][1]
        
        V_old = self.val_states.get((S_update,A_update), 0)
        self.val_states[(S_update,A_update)] += self.alpha * (G_t - V_old)

    def step_update(self, actions , state_last, rew_last):
        selected_action = self.policy(actions)
        self.episodes.append((state_last , selected_action ,  rew_last))

        if len(self.episodes) >= self.n:
            self.update()
        return selected_action