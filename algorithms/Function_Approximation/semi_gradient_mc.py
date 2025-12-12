import numpy as np

class SemiGradientMC:
    def __init__(
            self, 
            discount_rate, 
            alpha, 
            n,            # number of basis functions
            num_states,   # total number of states
            func_type='fourier'
        ):
        self.discount_rate = discount_rate
        self.n = n 
        self.alpha = alpha
        self.num_states = num_states
        self.w = np.zeros(n)  # initialize weights

        # set basis function
        if func_type == 'fourier':
            self.basis_fn = lambda state: np.array([np.cos(i * np.pi * state / num_states) for i in range(n)])
        elif func_type == 'poly':
            self.basis_fn = lambda state: (np.array([state / num_states]).reshape(1,-1) ** np.arange(n).reshape(-1,1)).squeeze()
        else:
            raise ValueError("func_type must be 'fourier' or 'poly'")

    def v_hat(self, state):
        """Predicted value for a state"""
        return np.dot(self.w, self.basis_fn(state))

    def update(self, episodes):
        """
        Update weights for one episode.
        episodes: list of tuples (state_id, reward), in order of visitation
        """
        G_t = 0
        # Reverse loop for correct MC returns
        for state_id, reward in reversed(episodes):
            G_t = reward + self.discount_rate * G_t
            # Semi-gradient MC update
            self.w += self.alpha * (G_t - self.v_hat(state_id)) * self.basis_fn(state_id)

    def estimate_v_mc(self):
        """Estimate value for all states using current weights"""
        values = np.zeros(self.num_states)
        for s in range(self.num_states):
            values[s] = self.v_hat(s)
        return values
