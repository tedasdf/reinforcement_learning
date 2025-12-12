import numpy as np
import matplotlib.pyplot as plt

# Example parameters
num_states = 10
left_window = 1
right_window = 1
start_state = 5

# ----------------------
# Build T
# ----------------------
T = np.zeros((num_states, num_states + 2))
i, j = np.indices(T.shape)

# uniform probability for neighbors
u = 1 / (left_window + right_window)
for k in range(1, right_window+1):
    T[i==j-k] = u
for k in range(left_window):
    T[i-k==j] = u

# handle edges (assign remaining probability to terminal states)
row_sums = np.sum(T, axis=1)
for k in range(T.shape[0]):
    if T[k,0] != 0:
        T[k,0] += 1 - row_sums[k]
    if T[k,-1] !=0:
        T[k,-1] += 1 - row_sums[k]

# Add rows for terminal states
T = np.vstack((np.zeros(T.shape[1]), T, np.zeros(T.shape[1])))
true_T = T.copy()
T[0,start_state] = 1
T[-1,start_state] = 1
true_T[0,0] = 1
true_T[-1,-1] = 1

# ----------------------
# Rewards
# ----------------------
rewards = np.zeros(num_states + 2)
rewards[0] = -1
rewards[-1] = 1

# ----------------------
# Visualization
# ----------------------
plt.figure(figsize=(10,5))
plt.imshow(true_T, cmap='viridis', interpolation='none')
plt.colorbar(label='Transition probability')
plt.title('True Transition Matrix (true_T)')
plt.xlabel('Next state')
plt.ylabel('Current state')
plt.show()

print("Rewards vector:")
print(rewards)
