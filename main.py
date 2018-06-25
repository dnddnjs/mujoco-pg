import gym
import torch
from model import Actor, Critic
from utils import get_action
from collections import deque
from trpo import train_actor, train_critic

# you can choose other environments.
# possible environments: Ant-v2, HalfCheetah-v2, Hopper-v2, Humanoid-v2,
# HumanoidStandup-v2, InvertedPendulum-v2, Reacher-v2, Swimmer-v2, Walker2D-v2
env = gym.make("Walker2d-v2")

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

print('state size:', num_inputs)
print('action size:', num_actions)

actor = Actor(num_inputs, num_actions)
critic = Critic(num_inputs)


for iter in range(1000):
    memory = deque()

    score = 0
    steps = 0

    while steps < 15000:
        state = env.reset()

        for _ in range(10000):
            env.render()
            steps += 1
            mu, std = actor(torch.Tensor(state))
            action = get_action(mu, std)
            next_state, reward, done, _ = env.step(action)

            if done:
                mask = 0
            else:
                mask = 1

            memory.append([state, action, reward, mask])

            score += reward
            state = next_state

            if done:
                break

    train_actor()
    train_critic()