from agent import Env, Agent
import numpy as np
from myKukaGymEnv import MyKukaGymEnv
from settings import argumentsRL
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
#import tensorflow as tf
from replay_buffer import ComBuffer, ReplayBuffer
import pybullet_data
import pybullet as p
from my_policy_gradient import GaussianPolicyTorch, ConvNet, FcNet, DiagGaussian
from torch.distributions import Normal
import random

urdfs = ['CatLying_800_tex.urdf', 'CoffeeCookies_800_tex.urdf', 'Tortoise_800_tex.urdf']

args = argumentsRL()
urdf_root = '/Users/maxgold/Documents/Princeton/Senior/thesis/code/mycode/urdfs/dexnet'
object_name = urdfs[2]
env = MyKukaGymEnv(renders=True, isDiscrete=False, 
									objUrdfRoot=urdf_root, objectName=object_name)
env = env.unwrapped
env.seed(1)
env.reset()



import matplotlib.pyplot as plt

env.x0 = .65
env.y0 = 0
env.xscale = 0.02
env.yscale = 0.02
env.angle_scale = 1

env.reset(500)


for obj in urdfs:
	env._objectName = obj
	print(obj)
	success = 0
	for _ in range(10):
		env.reset(500)
		action = env.getBlockPosition()[:2]
		action = (action[0] + 0, action[1])
		_, s, _, _ = env.step(action, graspSteps=500, liftSteps=500, downSteps=500)
		success += s
	print(obj, '  ', success)

num_obs = env.observation_space.shape[0]
num_act = env.action_space.shape[0]


dist_dim           = 6
args.action_layers = [num_obs, 20, 10, dist_dim]
args.obs_dim       = num_obs
args.action_dim    = num_act
args.gamma         = .99
args.com_embed_dim = 10
args.mlp_layers    = [10, args.com_embed_dim]
args.actor_lr      = .001
args.com_lr        = .001
args.com_mb_size   = 16
args.com_episodes  = 10000
args.com_updates   = 10
args.com_summary   = 10
args.momentum      = .9
args.buffer_size   = 1e6

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high
com_dim = 3

# trains well
env.x0 = .25
env.y0 = 0.2
env.xscale = 0.02
env.yscale = 0.02
env.angle_scale = 1

## NEED TO NORMALIZE THE INPUTS!!
env.x0 = .6
env.y0 = 0
env.xscale = 0.1
env.yscale = 0.1
env.angle_scale = 1
w = 480
h = 360
env.reset()
x = env.clean_render(width=w,height=h)
plt.imshow(x)
plt.show()

num_filter = 10
num_fc     = 480

com_net = ConvNet([3,5,num_filter],[num_fc,20,20,3])

#com_net = FcNet([state_dim, 20, 3])
actor = GaussianPolicyTorch(args.action_layers)
normal = DiagGaussian(dist_dim, num_act)

w = 480
h = 360
x = env.clean_render(width=w,height=h)
x = Variable(torch.FloatTensor(x).unsqueeze(0).permute(0,3,1,2))

com_net(x)

com_optimizer = optim.Adam(com_net.parameters(), lr=.001)	
actor_optimizer = optim.Adam(actor.parameters(), lr=.001)

com_buffer = ComBuffer(int(args.buffer_size), int(args.random_seed))
replay_buffer = ReplayBuffer(int(args.buffer_size), int(args.random_seed))

for i in range(int(args.com_episodes)):
	obj = random.choice(urdfs)
	env._objectName = obj
	env.reset(500)
	x = env.clean_render(width=w,height=h)
	x = torch.FloatTensor(x).unsqueeze(0).permute(0,3,1,2).numpy()
	com = np.array(env.getBlockPosition())
	com_buffer.add(x, np.reshape(com, (com_dim,)))

	if i % 100 == 0:
		print(i)
	cum_loss = 0
	if com_buffer.size() > 300:
		for j in range(args.com_updates):
			state_batch, com_batch = com_buffer.sample_batch(int(args.com_mb_size))
			state_batch, com_batch = Variable(torch.FloatTensor(state_batch)), Variable(torch.FloatTensor(com_batch))
			state_batch = state_batch.squeeze(1)
			com_optimizer.zero_grad()
			output = com_net(state_batch)
			loss = F.mse_loss(output, com_batch.float())
			loss.backward()
			com_optimizer.step()
			cum_loss += np.sqrt(loss.data[0])

		if i % args.com_summary == 0:
			print('| Batch loss: {:.4f} | Update step: {:d}'.format(cum_loss/args.com_updates, i))



log_prob_mu = 0
successes = 0
for i in range(1000):
	obj = random.choice(urdfs)
	env._objectName = obj
	s   = env.reset(500)
	s   = Variable(torch.FloatTensor(s))
	x = env.clean_render(width=w,height=h)
	x = torch.FloatTensor(x).unsqueeze(0).permute(0,3,1,2)
	x = Variable(x)

	out  = actor(s)

	action_dist = normal(out)
	action = normal.sample(out, deterministic=False)
	action_log_probs, action_entropy = normal.logprobs_and_entropy(out, action)

	true_com = np.array(env.getBlockPosition())

	com    = com_net(x)
	adjust_action      = (0 + com).data.numpy()[0]
	a = (adjust_action[0], adjust_action[1])

	rmse = np.sqrt(np.sum((true_com - com.data.numpy())**2))
	#print(rmse)

	_, r, _, _ = env.step(a, graspSteps=500, liftSteps=500, downSteps=500)
	if r == 0:
		print('Failure on ', obj)
	else:
		successes += 1
		
	replay_buffer.add(np.reshape(s, (actor.state_dim,)), np.reshape(a, (2,)), 
								action_log_probs, r)



	loss     = - action_log_probs * r
	loss.backward()
	actor_optimizer.step()
	log_prob_mu += action_log_probs.data.numpy()[0][0]
	if i % 10 == 0:
		print('Average log prob: ', log_prob_mu/10)
		print("Success rate: {0:.3f}".format(successes/(i+1)))
		log_prob_mu = 0


def one_step():
	obj = random.choice(urdfs)
	env._objectName = obj
	s = env.reset(500)
	s2 = s

	s = Variable(torch.FloatTensor(s))

	out  = actor(s)

	action_dist = normal(out)
	action = normal.sample(out, deterministic=False)
	action_log_probs, action_entropy = normal.logprobs_and_entropy(out, action)

	com    = com_net(s)
	com    = Variable(torch.FloatTensor(env.getBlockPosition()))
	adjust_action      = (0 + com).data.numpy()
	a = (adjust_action[0], adjust_action[1])

	_, r, _, _ = env.step(a, graspSteps=500, liftSteps=500, downSteps=500)

	replay_buffer.add(np.reshape(s, (actor.state_dim,)), np.reshape(a, (2,)), 
								action_log_probs, r)



	loss     = - action_log_probs * r
	loss.backward()
	actor_optimizer.step()








