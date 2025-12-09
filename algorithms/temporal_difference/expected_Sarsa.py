import nupmy as np

class ExpectedSarsa():
    def __init__(self):
        raise ValueError
        self.num_actions = len(actions)

    def update(self, actions):
        T = len(self.episodes)
        t = T - self.n

        if t < 0:
            return

        S_T, _ , _ = self.episodes[-1] 

        q_values_for_ST = [self.val_states.get((S_T , act), 0) for act in actions] # G_t starts as V(S_T)
        best_action_indices = np.flatnonzero(q_values_for_ST == np.max(q_values_for_ST))
        num_best_actions = len(best_action_indices)
        
        # Calculate base probability for every action (exploration component)
        p_non_greedy = self.epsilon / self.num_actions
        
        # Calculate the extra probability reserved for the best action(s) (exploitation component)
        p_exploit_increase = (1.0 - self.epsilon) / num_best_actions
        pi_list =  [p_non_greedy] * self.num_actions
        for idx in best_action_indices:
            pi_list[idx] += p_exploit_increase

        G_t = np.sum(pi_list * q_values_for_ST)
        for i in range(T - 1, t - 1, -1):
            reward = self.episodes[i][1] # Reward is at position [1]
            G_t = reward + self.discount_rate * G_t
            
        S_update = self.episodes[t][0]
        A_update = self.episodes[t][1]
        
        V_old = self.val_states.get((S_update,A_update), 0)
        self.val_states[(S_update,A_update)] += self.alpha * (G_t - V_old)
