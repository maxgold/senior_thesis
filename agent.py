from my_policy_gradient import tfPolicyGradient, torchPolicyGradient
import gym
import pybullet as p
from myKukaGymEnv import MyKukaGymEnv
from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from torch.autograd import Variable
import torch
from distributions import DiagonalGaussian
import numpy as np

class Env(gym.Env):
    def __init__(self, env_name, render=False, discrete=True, objectName = None):
        if env_name == 'kuka':
            self.env = KukaGymEnv(renders=render, isDiscrete=discrete, objectName=objectName)
            self.env = self.env.unwrapped
        elif env_name == 'cartpole':
            self.env = gym.make('CartPole-v0')
            self.env = self.env.unwrapped
        elif env_name == 'mykuka':
            self.env = MyKukaGymEnv
    @property
    def observation_space(self):
        return(self.env.observation_space)
    @property
    def action_space(self):
        return(self.env.action_space)
    def seed(self, rs):
        self.env.seed(rs)
    def render(self,mode):
        return self.env.render(mode)
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)


class Agent():
    def __init__(self, actor):
        #self.pg_type = pg_type
        #if pg_type == 'tf_PG':
        #    self.policy = tfPolicyGradient(args)
        #elif pg_type == 'torch_PG':
        #    self.policy = torchPolicyGradient(args)
        self.policy = actor
    def step_env(self, env, observation):
        import IPython as ipy
        #ipy.embed()

        action = self.policy.choose_action(observation)
        #action = int(action)

        observation_, reward, done, info = env.step(action.reshape(-1,), maxSteps=1000)
        self.policy.store_transition(observation, action, reward)

        return(observation_, done)

    def sim_episode(self, env, num_rollouts=1, max_steps=100, update=True):
        for _ in range(num_rollouts):
            observation = env.reset()
            episode_reward = 0

            for t in range(max_steps):
                observation, done = self.step_env(env, np.reshape(observation, (1,-1)))
                if done:
                    break
        
        episode_rewards_sum = sum(self.policy.episode_rewards)/num_rollouts
        
        if update:
            self.policy.learn()

        return(episode_rewards_sum)

    def enjoy_episode(self, env, max_steps=100):
        observation = env.reset()
        for t in range(max_steps):
            env.render(mode='rgb_array')
            observation, done = self.step_env(env, observation)
            if done:
                break 


def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        if is_atari:
            env = wrap_deepmind(env)
        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = WrapPyTorch(env)
        return env

    return _thunk


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [obs_shape[2], obs_shape[1], obs_shape[0]]
        )

    def _observation(self, observation):
        return observation.transpose(2, 0, 1)














