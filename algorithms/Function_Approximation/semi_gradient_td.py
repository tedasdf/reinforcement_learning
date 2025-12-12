import numpy as np

class SemiGradientTD:
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

    def td0_update_episode(self, episode):
        """
        Perform semi-gradient TD(0) updates for one episode.
        episode: list of tuples [(state, reward), ...]
        """
        for t in range(len(episode)):
            S_t, R_tplus1 = episode[t]
            # terminal state check
            if t == len(episode) - 1:
                S_next = None
            else:
                S_next = episode[t+1][0]

            # compute TD error
            if S_next is None:
                delta = R_tplus1 - self.v_hat(S_t)  # no bootstrapping for terminal
            else:
                delta = R_tplus1 + self.discount_rate * self.v_hat(S_next) - self.v_hat(S_t)

            # update weights
            self.w += self.alpha * delta * self.basis_fn(S_t)

    def estimate_v(self):
        """Estimate value for all states using current weights"""
        values = np.zeros(self.num_states)
        for s in range(self.num_states):
            values[s] = self.v_hat(s)
        return values
