

from mc_epsilon_greedy import MC_epsilon_greedy


class OffpolicyMC(MC_epsilon_greedy):
    def __init__(
            self,
            states,
            epsilon_pi,
            epsilon_b,
            
                      ):
        super().__init__(

        )
        self.b = MC_epsilon_greedy()
    
        self.epsilon = epsilon_pi
        self.pi_dict = {
            state: None for state in states
        }   
        

    def final_policy(self, actions):
        for state in self.pi_dict.keys():
            self.pi_dict[state] = self.get_best_action(state, actions)
        return self.pi_dict

    def update(self, episodes, actions):
        G_t = 0
        W = 1
        for states_id , action, reward in reversed(episodes):
            G_t = reward + self.discount_rate * G_t
            pi_action = self.get_best_action(states_id, actions) 

            if action != pi_action:
                break
            
            if (states_id , action) in self.val_states.keys():
                self.val_states[(states_id , action)] += self.alpha * ( G_t * W  -  self.val_states[(states_id , action)])
            
            W /= (1 - self.b.epsilon + self.b.epsilon/2)
        
    