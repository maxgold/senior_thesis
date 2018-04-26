from my_policy_gradient import MyPolicyGradient
#from pybullet_utils import *
import math
import gym
import numpy as np
import pybullet as p
import pybullet_data
import pickle
from kukaGymEnv import KukaGymEnv
from kukaCamGymEnv import KukaCamGymEnv
from kuka_diverse_object_gym_env import KukaDiverseObjectEnv


RENDER_ENV = True

env = KukaGymEnv(renders=RENDER_ENV, isDiscrete=True)
env = env.unwrapped

# Policy gradient has high variance, seed for reproducability
env.seed(1)

print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)
print("env.observation_space.high", env.observation_space.high)
print("env.observation_space.low", env.observation_space.low)


EPISODES = 500
rewards = []

PG = MyPolicyGradient(
   n_obs = env.observation_space.shape[0],
   n_act = env.action_space.n,
   learning_rate=0.01,
   discount=0.99,
   hidden_units=[10],
   scope='1'
)


rgb_array = env.render(mode='rgb_array')
EPISODES=1000
for episode in range(EPISODES):

    observation = env.reset()
    blockPos = (.6,.15,-.186)
    blockOrn = (0,0,.98,.14)
    p.resetBasePositionAndOrientation(env.blockUid,blockPos,blockOrn)
    episode_reward = 0
    steps = 0
    while True:
        #if RENDER_ENV: env.render()
        steps += 1

        # 1. Choose an action based on observation
        action = PG.choose_action(observation)

        # 2. Take action in the environment
        observation_, reward, done, info = env.step(action)

        # 3. Store transition for training
        PG.store_transition(observation, action, reward)
        
        dist = env.dist_to_block()
        #print("Final dist:", dist)

        if done:
            episode_rewards_sum = sum(PG.episode_rewards)
            rewards.append(episode_rewards_sum)
            max_reward_so_far = np.amax(rewards)

            print("==========================================")
            print("Episode: ", episode)
            print("Reward: ", episode_rewards_sum)
            print("Max reward so far: ", max_reward_so_far)
            print("Steps taken:", steps)
            print("Min dist:", max(PG.episode_rewards))

            discounted_episode_rewards_norm = PG.learn()
            print("Final dist:", dist)


            break

        # Save new observation
        observation = observation_
        RENDER_ENV=False





