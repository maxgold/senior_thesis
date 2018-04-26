import os
import time

import gym
from gym.spaces.box import Box

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from dataHelper import RolloutStorage
from agent import make_env
from model import MLPPolicy



args = argumentsRL()

envs = [make_env('CartPole-v0', args.seed, i, None)
                for i in range(args.num_processes)]

if args.num_processes == 1:
	envs = DummyVecEnv(envs)
else:
	envs = SubprocVecEnv(envs)

if len(envs.observation_space.shape) == 1:
    envs = VecNormalize(envs)

obs_shape = envs.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])




if len(envs.observation_space.shape) == 3:
    actor_critic = CNNPolicy(obs_shape[0], envs.action_space, args.recurrent_policy)
else:
    assert not args.recurrent_policy, \
        "Recurrent policy is not implemented for the MLP controller"
    actor_critic = MLPPolicy(obs_shape[0], envs.action_space)

if envs.action_space.__class__.__name__ == "Discrete":
    action_shape = 1
else:
    action_shape = envs.action_space.shape[0]


if args.algo == 'a2c':
    optimizer = optim.RMSprop(actor_critic.parameters(), args.learning_rate)
elif args.algo == 'ppo':
    optimizer = optim.Adam(actor_critic.parameters(), args.learning_rate)
    
rollouts    = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size)
current_obs = torch.zeros(args.num_processes, *obs_shape)

def update_current_obs(obs):
    shape_dim0 = envs.observation_space.shape[0]
    obs = torch.from_numpy(obs).float()
    if args.num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs

obs = envs.reset()
update_current_obs(obs)

rollouts.observations[0].copy_(current_obs)

# These variables are used to compute average rewards for all processes.
episode_rewards = torch.zeros([args.num_processes, 1])
final_rewards   = torch.zeros([args.num_processes, 1])


start = time.time()
for j in range(args.num_updates):
    for step in range(args.num_steps):
        # Sample actions
        value, action, action_log_prob, states = actor_critic.act(Variable(rollouts.observations[step], volatile=True),
                                                                  Variable(rollouts.states[step], volatile=True),
                                                                  Variable(rollouts.masks[step], volatile=True))
        cpu_actions = action.data.squeeze(1).cpu().numpy()

        # Observe reward and next obs
        obs, reward, done, info = envs.step(cpu_actions)
        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
        episode_rewards += reward

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        final_rewards *= masks
        final_rewards += (1 - masks) * episode_rewards
        episode_rewards *= masks

        if current_obs.dim() == 4:
            current_obs *= masks.unsqueeze(2).unsqueeze(2)
        else:
            current_obs *= masks

        update_current_obs(obs)
        rollouts.insert(step, current_obs, states.data, action.data, action_log_prob.data, value.data, reward, masks)

    next_value = actor_critic(Variable(rollouts.observations[-1], volatile=True),
                              Variable(rollouts.states[-1], volatile=True),
                              Variable(rollouts.masks[-1], volatile=True))[0].data

    rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

    if args.algo == 'a2c':
        values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(Variable(rollouts.observations[:-1].view(-1, *obs_shape)),
                                                                                       Variable(rollouts.states[0].view(-1, actor_critic.state_size)),
                                                                                       Variable(rollouts.masks[:-1].view(-1, 1)),
                                                                                       Variable(rollouts.actions.view(-1, action_shape)))

        values = values.view(args.num_steps, args.num_processes, 1)
        action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)

        advantages = Variable(rollouts.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(Variable(advantages.data) * action_log_probs).mean()

        optimizer.zero_grad()
        (value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()

        nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

        optimizer.step()

    elif args.algo == 'ppo':
    	raise NotImplementedError

    rollouts.after_update()

    if j % args.log_interval == 0:
        end = time.time()
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.3f}, value loss {:.3f}, policy loss {:.3f}".
            format(j, total_num_steps,
                   int(total_num_steps / (end - start)),
                   final_rewards.mean(),
                   final_rewards.median(),
                   final_rewards.min(),
                   final_rewards.max(), dist_entropy.data[0],
                   value_loss.data[0], action_loss.data[0]))

        lengths = []
        for i in range(10):
            tracker = np.zeros((0,args.num_processes))
            obs = envs.reset()
            for t in range(500):
                value, action, action_log_prob, states = actor_critic.act(Variable(torch.Tensor(obs), volatile=True), None, None)
                action = action.data.squeeze(1).cpu().numpy()
                obs, reward, done, info = envs.step(action)
                tracker = np.r_[tracker, done[None,:]]

            failures = np.where(tracker == 1)
            seen = []
            current_lengths = np.zeros((args.num_processes))
            for (t_steps, proc) in zip(failures[0], failures[1]):
                if proc not in seen:
                    current_lengths[proc] = t_steps
                    seen.append(proc)
            lengths.append(current_lengths)
        lengths = np.vstack(lengths)

        print("Interpretable mean reward {:.1f}, min/max reward {:.1f}/{:.1f}".format(np.mean(lengths), np.min(lengths), np.max(lengths)))
        





def run_episode(actor_critic, rollouts, args):
    simulate_steps(actor_critic, rollouts)
    next_value = actor_critic(Variable(rollouts.observations[-1], volatile=True),
                              Variable(rollouts.states[-1], volatile=True),
                              Variable(rollouts.masks[-1], volatile=True))[0].data

    rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

    update_params(actor_critic, rollouts, optimizer, args)


def simulate_steps(actor_critic, rollouts):
    for step in range(num_steps):
        # Sample actions
        value, action, action_log_prob, states = actor_critic.act(Variable(rollouts.observations[step], volatile=True),
                                                                  Variable(rollouts.states[step], volatile=True),
                                                                  Variable(rollouts.masks[step], volatile=True))
        cpu_actions = action.data.squeeze(1).cpu().numpy()

        # Obser reward and next obs
        obs, reward, done, info = envs.step(cpu_actions)
        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
        episode_rewards += reward

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        final_rewards *= masks
        final_rewards += (1 - masks) * episode_rewards
        episode_rewards *= masks

        if current_obs.dim() == 4:
            current_obs *= masks.unsqueeze(2).unsqueeze(2)
        else:
            current_obs *= masks

        update_current_obs(obs)
        rollouts.insert(step, current_obs, states.data, action.data, action_log_prob.data, value.data, reward, masks)






