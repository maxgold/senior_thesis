from agent import Env, Agent
import numpy as np
from myKukaGymEnv import MyKukaGymEnv
from settings import argumentsRL
from actor_critic_network import *
from replay_buffer import ReplayBuffer, ComBuffer
import tensorflow as tf


args = argumentsRL()

env = MyKukaGymEnv(renders=False, isDiscrete=False)
env = env.unwrapped
env.seed(1)
env.reset()

env.reset()
pos = env.getBlockPosition()
env.step((pos[0], pos[1]), maxSteps=1000)


num_obs = env.observation_space.shape[0]
num_act = env.action_space.shape[0]


args.action_layers = [40, 30]
args.com_embed_dim = 10
args.mlp_layers    = [10, args.com_embed_dim]
args.obs_dim       = num_obs
args.action_dim    = num_act
args.gamma         = .99

args.actor_lr        = .001
args.critic_lr       = .001
args.mlp_lr          = .001
args.gamma           = .99
args.tau             = .001
args.buffer_size     = 1000000
args.mb_size         = 16
args.com_mb_size     = 16
args.random_seed     = 1234
args.max_episodes    = 1000
args.max_episode_len = 5
args.num_updates     = 1
args.com_episodes    = 1250
args.com_updates     = 10
args.com_summary     = 25


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high
com_dim = 3


sess = tf.Session()

actor = ActorNetwork(sess, args.action_layers, state_dim, action_dim, action_bound,
                     float(args.actor_lr), float(args.tau),
                     int(args.mb_size))

critic = CriticNetwork(sess, state_dim, action_dim,
                       float(args.critic_lr), float(args.tau),
                       float(args.gamma),
                       actor.get_num_trainable_vars())

com_net = MLPNetwork(sess, args.mlp_layers, state_dim, com_dim, args.mlp_lr, args.com_mb_size)


#actor_noise = lambda : 0
actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))


sess.run(tf.global_variables_initializer())

com_buffer = ComBuffer(int(args.buffer_size), int(args.random_seed))

for i in range(int(args.com_episodes)):
    s = env.reset()
    com = np.array(env.getBlockPosition())
    com_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(com, (com_dim,)))

    loss = 0
    if com_buffer.size() > int(args.com_mb_size):
          for j in range(args.com_updates):
            s_batch, com_batch = com_buffer.sample_batch(int(args.com_mb_size))

          # Update the critic given the targets
            com_net.train(s_batch, com_batch)
            loss += com_net.eval_loss(s_batch, com_batch)

          if i % args.com_summary == 0:
            print('| Batch loss: {:.4f} | Update step: {:d}'.format(loss/args.com_updates, i))


for i in range(int(args.max_episodes)):
    s = env.reset()

    ep_reward = 0
    ep_ave_max_q = 0

    for j in range(int(args.max_episode_len)):

        if args.render_env:
            env.render()

        # Added exploration noise
        com = com_net.predict(s.reshape(-1, state_dim))
        a_pred = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()

        a_pred = np.clip(a_pred, -actor.action_bound, actor.action_bound)
        #print(a)
        #com = env.getBlockPosition()

        a = com[0][:2] + a_pred[0]
        pos = env.getBlockPosition()
        dist = np.sqrt(sum([(pos[i]  - a[i])**2 for i in range(2)]))

        s2, r, terminal, info = env.step(a, maxSteps=1000)

        replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a_pred, (actor.a_dim,)), r,
                          terminal, np.reshape(s2, (actor.s_dim,)))


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
            print('| Reward: {:.4f} | Episode: {:d} | Qmax: {:.4f} | Dist from center : {:.3f}'.format(ep_reward, \
                    i, (ep_ave_max_q / (float(j+1) * args.num_updates)), dist))
            break




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
        com = com_net.predict(s.reshape(-1, state_dim))
        a_pred = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()

        a_pred = np.clip(a_pred, -actor.action_bound, actor.action_bound)
        #print(a)
        #com = env.getBlockPosition()

        a = com[0][:2] + a_pred[0]
        pos = env.getBlockPosition()
        dist = np.sqrt(sum([(pos[i]  - a[i])**2 for i in range(2)]))

        s2, r, terminal, info = env.step(a, maxSteps=1000)

        replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a_pred, (actor.a_dim,)), r,
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
            print('| Reward: {:.4f} | Episode: {:d} | Qmax: {:.4f} | Dist from center : {:.3f}'.format(ep_reward, \
                    i, (ep_ave_max_q / (float(j+1) * args.num_updates)), dist))
            break


import numpy as np
import matplotlib.pyplot as plt
modes = ['full', 'same', 'valid']
N = 30
    
plt.plot(np.convolve(rewards, np.ones((N,))/N, mode='valid'))
plt.show()


for i in range(25):
  action = np.zeros(25)
  action[i] = 1
  observation, reward, done, info = env.step(action)
  print(reward)












