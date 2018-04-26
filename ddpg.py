import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import argparse
import pprint as pp
import random
from actor_critic_network import *
from replay_buffer import ReplayBuffer
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic, actor_noise):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    if args.write_summary:
        writer = tf.summary.FileWriter(args.summary_dir, sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args.buffer_size), int(args.random_seed))

    for i in range(int(args.max_episodes)):
        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args.max_episode_len)):

            if args.render_env:
                env.render()

            # Added exploration noise
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()
            a = np.clip(a, -actor.action_bound, actor.action_bound)


            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args.mb_size):
                for _ in range(args.num_updates):
                    s_batch, a_batch, r_batch, t_batch, s2_batch = \
                        replay_buffer.sample_batch(int(args.mb_size))

                    # Calculate targets
                    target_q = critic.predict_target(
                        s2_batch, actor.predict_target(s2_batch))

                    y_i = []
                    for k in range(int(args.mb_size)):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + critic.gamma * target_q[k])

                    # Update the critic given the targets
                    predicted_q_value, _ = critic.train(
                        s_batch, a_batch, np.reshape(y_i, (int(args.mb_size), 1)))

                    ep_ave_max_q += np.amax(predicted_q_value)

                    # Update the actor policy using the sampled gradient
                    a_outs = actor.predict(s_batch)
                    grads = critic.action_gradients(s_batch, a_outs)
                    actor.train(s_batch, grads[0])

                    # Update target networks
                    actor.update_target_network()
                    critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                if args.write_summary:
                    writer.add_summary(summary_str, i)
                    writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} | Episode length : {:d}'.format(int(ep_reward), \
                        i, (ep_ave_max_q / (float(j) * args.num_updates)), j))
                break

def main(args):
    sess = tf.Session()

    #env = gym.make('Pendulum-v0')
    #env = gym.make('MountainCarContinuous-v0')
    #env = gym.make('LunarLanderContinuous-v2')
    #env = gym.make('BipedalWalker-v2')
    env = KukaGymEnv(isDiscrete=False, renders=False)

    env._max_episode_steps = args.max_episode_len
    np.random.seed(int(args.random_seed))
    tf.set_random_seed(int(args.random_seed))
    env.seed(int(args.random_seed))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high
    # Ensure action bound is symmetric
    assert np.all(env.action_space.high == -env.action_space.low)

    actor = ActorNetwork(sess, args.action_layers, state_dim, action_dim, action_bound,
                         float(args.actor_lr), float(args.tau),
                         int(args.mb_size))

    critic = CriticNetwork(sess, state_dim, action_dim,
                           float(args.critic_lr), float(args.tau),
                           float(args.gamma),
                           actor.get_num_trainable_vars())
    
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    if args.use_gym_monitor:
        if not args.render_env:
            env = wrappers.Monitor(
                env, args.monitor_dir, video_callable=False, force=True)
        else:
            env = wrappers.Monitor(env, args.monitor_dir, force=True)

    train(sess, env, args, actor, critic, actor_noise)

    if args.use_gym_monitor:
        env.monitor.close()




if __name__ == '__main__':
    from settings import argumentsRL
    args = argumentsRL()
    args.actor_lr        = .001
    args.critic_lr       = .001
    args.gamma           = .99
    args.tau             = .001
    args.buffer_size     = 1000000
    args.mb_size         = 64
    args.random_seed     = 1234
    args.max_episodes    = 20000
    args.max_episode_len = 5000
    args.num_updates     = 1

    args.action_layers   = [400, 300]

    main(args)


    
















