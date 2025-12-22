'''
Docstring for rl_project.buffer.prioritised_replay
https://arxiv.org/pdf/1511.05952
'''

import random 
import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # sum tree
        self.data = np.zeros(capacity, dtype=object)  # store transitions
        self.ptr = 0
        self.n_entries = 0

    def add(self, priority, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        # propagate the change up
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def sample(self, s):
        idx = 0
        while idx < self.capacity - 1:  # while not leaf
            left = 2 * idx + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self):
        return self.tree[0]

class PrioritizedReplayBuffer():
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, eps=1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha      # priority exponent
        self.beta = beta        # IS weight exponent
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.eps = eps
        self.max_priority = 1.0 # initial max priority


    def store(self,transition):
        """Stores a transition tuple in the buffer."""
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total / batch_size
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])  # anneal beta

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.sample(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        probs = np.array(priorities) / self.tree.total
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        weights /= weights.max()  # normalize
        weights = np.array(weights, dtype=np.float32)

        return idxs, batch, weights
    
    def update_priorities(self, idxs, td_errors):
        """Update priorities after learning."""
        for idx, td_error in zip(idxs, td_errors):
            priority = (abs(td_error) + self.eps) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

if __name__ == "__main__":
    buffer = PrioritizedReplayBuffer(capacity=10)

    # store some fake transitions
    for i in range(10):
        s = np.random.randn(4)
        a = np.random.randint(0, 2)
        r = np.random.randn()
        s_ = np.random.randn(4)
        d = np.random.randint(0,2)
        buffer.store((s, a, r, s_, d))

    # sample a batch
    idxs, batch, weights = buffer.sample(batch_size=5)
    print("Sampled indices:", idxs)
    print("Sampled transitions:", batch)
    print("Importance weights:", weights)

    # fake TD errors
    td_errors = np.random.randn(5)
    buffer.update_priorities(idxs, td_errors)
