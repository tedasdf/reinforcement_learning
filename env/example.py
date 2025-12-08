import gym
import gym_chess
import random

env = gym.make('Chess-v0')
print(env.render())

env.reset()
done = False

while not done:
    action = random.sample(env.legal_moves,1)[0]
    env.step(action)
    print(env.render(mode='unicode'))

env.close()