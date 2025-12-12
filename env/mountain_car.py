import math

class MountainCar():
    def __init__(self):
        
        # state of the car
        self.position = None
        self.velocity = None

        # constraint
        self.min_pos = -1.2 
        self.max_pos = 0.6
        self.max_velocity = 0.07

        # goal position
        self.goal_pos = 0.5

        # env 
        self.force = 0.001
        self.gravity = 0.0025

    def reset(self):
        """Reset environment to initial state."""
        self.position = -0.5
        self.velocity = 0.0
        return (self.position, self.velocity)

    def step(self, action):
        """
        Take one action.
        action ∈ {0: push_left, 1: no_push, 2: push_right}
        
        Returns:
            next_state: (position, velocity)
            reward: float
            terminated: bool
            info: dict
        """

        # Convert discrete action into force [-1, 0, +1]
        act = action - 1  # map: 0→-1, 1→0, 2→+1

        # Update velocity (same equation as your C++ code)
        self.velocity += act * self.force + math.cos(3 * self.position) * (-self.gravity)

        # Clip velocity
        self.velocity = max(-self.max_velocity, min(self.velocity, self.max_velocity))

        # Update position
        self.position += self.velocity

        # Clip position
        if self.position < self.min_pos:
            self.position = self.min_pos
            if self.velocity < 0:
                self.velocity = 0

        if self.position > self.max_pos:
            self.position = self.max_pos

        # Reward
        reward = -1.0

        # Terminal condition
        terminated = self.position >= self.goal_pos

        return (self.position, self.velocity), reward, terminated, {}
