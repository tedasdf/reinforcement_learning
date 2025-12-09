 

from temporal_difference.n_step_TD_control import n_step_Sarsa


class Qlearning(n_step_Sarsa):
    def __init__(self):
        super().__init__(
        )

    def update(self, actions):
        T = len(self.episodes)
        t = T - self.n

        if t < 0:
            return

        S_T, _ , _ = self.episodes[-1] 

        q_values_for_ST = [self.val_states.get((S_T , act), 0) for act in actions] # G_t starts as V(S_T)
        G_t = max(q_values_for_ST)
        for i in range(T - 1, t - 1, -1):
            reward = self.episodes[i][1] # Reward is at position [1]
            G_t = reward + self.discount_rate * G_t
            
        S_update = self.episodes[t][0]
        A_update = self.episodes[t][1]
        
        V_old = self.val_states.get((S_update,A_update), 0)
        self.val_states[(S_update,A_update)] += self.alpha * (G_t - V_old)
