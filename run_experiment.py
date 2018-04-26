from agent import Env, Agent
import numpy as np
from myKukaGymEnv import MyKukaGymEnv
from settings import argumentsRL
import torch
from torch.autograd import Variable
import tensorflow as tf
from replay_buffer import ComBuffer
import pybullet_data
import pybullet as p
from my_policy_gradient import GaussianNetwork, MLPNetwork, ConvNet


urdfs = ['CatLying_800_tex.urdf', 'CoffeeCookies_800_tex.urdf', 'Glassbowl_800_tex.urdf', 
        'MilkDrinkVanilla_800_tex.urdf', 'OrangeMarmelade_800_tex.urdf', 'Peanuts2_800_tex.urdf',
        'Sauerkraut_800_tex.urdf', 'Seal_800_tex.urdf', 'Tortoise_800_tex.urdf']

args = argumentsRL()
#env = Env('cartpole')
#urdf_root = pybullet_data.getDataPath()
#object_name = 'block.urdf'
urdf_root = '/Users/maxgold/Documents/Princeton/Senior/thesis/code/mycode/urdfs/dexnet'
object_name = urdfs[0]
env = MyKukaGymEnv(renders=False, isDiscrete=False, 
                  objUrdfRoot=urdf_root, objectName=object_name)
env = env.unwrapped
env.seed(1)
env.reset()


for obj in urdfs:
  env._objectName = obj
  print(obj)
  success = 0
  for _ in range(10):
    env.reset()
    for _ in range(1000):
      p.stepSimulation()
    action = env.getBlockPosition()[:2]
    action = (action[0] + 0, action[1]-.02)
    _, s, _, _ = env.step(action, graspSteps=500, liftSteps=500, downSteps=500)
    success += s
  print(obj, '  ', success)

num_obs = env.observation_space.shape[0]
#num_act = env.action_space.n
num_act = env.action_space.shape[0]
#num_act = env.grid.shape[0]**2


num_filter = 10
num_fc     = num_filter*16

model = ConvNet([1,5,num_filter],[num_fc,10,2])




args.action_layers = [num_obs, 20, 10]
args.obs_dim       = num_obs
args.action_dim    = num_act
args.gamma         = .99
args.com_embed_dim = 10
args.mlp_layers    = [10, args.com_embed_dim]
args.actor_lr        = .001
args.mlp_lr          = .001
args.com_mb_size     = 16
args.com_episodes    = 1250
args.com_updates     = 10
args.com_summary     = 25

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high
com_dim = 3


sess = tf.Session()

com_net = MLPNetwork(sess, args.mlp_layers, state_dim, com_dim, args.mlp_lr, args.com_mb_size)
actor = GaussianNetwork(sess, args.action_layers, state_dim, action_dim, 
                        args.actor_lr, args.gamma, args.mb_size, com_net=com_net)
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


env = MyKukaGymEnv(renders=True, isDiscrete=False, 
                  objUrdfRoot=urdf_root, objectName=object_name)


agent = Agent(actor)

obs = env.reset()
actor.choose_action(obs.reshape(1,12))


#rgb_array = env.render(mode='rgb_array')
EPISODES= 1000
rewards  = []
max_steps = 1000
RENDER_ENV = False
num_rollouts= 10

episode_reward = agent.sim_episode(env, num_rollouts=num_rollouts, max_steps=max_steps)


for episode in range(EPISODES):
    episode_reward = agent.sim_episode(env, num_rollouts=num_rollouts, max_steps=max_steps)
    rewards.append(episode_reward)
    max_reward = np.amax(rewards)
    print("==========================================")
    print("Episode: ", episode)
    print("Reward: ", episode_reward)
    print("Max reward so far: ", max_reward)


agent.enjoy_episode(env, 10000)

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












