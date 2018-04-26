import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.distributions import Categorical
import tflearn
import math
import functools
import operator

class torchPolicyGradient(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.num_action_layers = len(args.action_layers)-1
		self.num_value_layers = len(args.value_layers)-1
		action_layers = args.action_layers
		value_layers  = args.value_layers
		self.action_model = nn.ModuleList([nn.Linear(action_layers[i],action_layers[i+1]) for i in range(self.num_action_layers)])
		self.value_model = nn.ModuleList([nn.Linear(value_layers[i],value_layers[i+1]) for i in range(self.num_value_layers)])
		
		for layer in self.action_model:
			torch.nn.init.xavier_uniform(layer.weight)

		self.episode_rewards = []
		self.episode_obs     = []
		self.episode_actions = []
		self.gamma = args.gamma
		self.lr    = args.lr
		self.optimizer = optim.Adam(self.parameters(), args.lr)

	def choose_action(self, observation, deterministic=False):

		action_probs = self(observation)
		m = Categorical(action_probs)
		actions = m.sample()

		return(actions)
	
	def discount_rewards(self):
		running_mean = 0
		discounted_rewards = np.zeros(len(self.episode_rewards))
		for i in reversed(range(len(self.episode_rewards))):
			running_mean = running_mean * self.gamma + self.episode_rewards[i]
			discounted_rewards[i] = running_mean

		discounted_rewards -= np.mean(discounted_rewards)
		discounted_rewards /= np.std(discounted_rewards)
		return(discounted_rewards)

	def learn(self):
		values, action_log_probs, dist_entropy, states = self.evaluate_actions()

		#values           = values.view(len(self.episode_rewards),1)
		#action_log_probs = action_log_probs.view(len(self.episode_rewards), 1)

		normalized_returns = self.discount_rewards()

		advantages = Variable(torch.Tensor(normalized_returns))
		#advantages = Variable(normalized_returns) - values
		#value_loss = advantages.pow(2).mean()

		action_loss = -(Variable(advantages.data) * action_log_probs).mean()

		self.zero_grad()

		self.optimizer.zero_grad()
		#(value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()
		(action_loss).backward()

		if self.args.clip_grad_norm:
			nn.utils.clip_grad_norm(self.parameters(), args.max_grad_norm)

		self.optimizer.step()
		self.episode_rewards = []
		self.episode_obs     = []
		self.episode_actions = []


	def store_transition(self, o, a, r):
		self.episode_obs.append(o)
		self.episode_rewards.append(r)
		action_onehot = np.zeros(self.args.action_dim)
		action_onehot[a] = 1
		self.episode_actions.append(action_onehot)
	
	def evaluate_actions(self):
		inputs  = Variable(torch.Tensor(np.vstack(self.episode_obs)))
		actions = Variable(torch.Tensor(np.vstack(self.episode_actions)))
		action_probs = self(inputs)
		#import IPython as ipy
		#ipy.embed()
		m = Categorical(action_probs)
		actions = m.sample()
		action_log_probs = m.log_prob(actions)
		#value_preds = self.value_model(inputs)
		value_preds = None
		states      = None

		return value_preds, action_log_probs, 0, states

	def forward(self, x):
		x = x.view(-1, self.args.obs_dim)
		for i in range(self.num_action_layers-1):
			x = F.relu(self.action_model[i](x))
		x = self.action_model[-1](x)
		return F.softmax(x, dim=-1)

	@property
	def num_parameters(self):
		num_p = 0
		for p in self.parameters():
			num_p += functools.reduce(operator.mul, p.size(), 1)
		return(num_p)


class tfPolicyGradient:
	def __init__(self, args):
		self.n_obs = args.obs_dim
		self.n_act = args.action_dim
		self.lr = args.lr
		self.gamma = args.gamma
		self.episode_rewards = []
		self.episode_obs     = []
		self.episode_actions = []
		self.scope = 'test'
		self.build_network(args.action_layers[1:], network='MLP')
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def build_network(self, hidden_units, network = 'MLP'):
		if network == 'MLP':
			with tf.name_scope('inputs' + self.scope):
				self.X = tf.placeholder(tf.float32, shape=(self.n_obs, None), name="X")
				self.Y = tf.placeholder(tf.float32, shape=(self.n_act, None), name="Y")
				self.training_rewards = tf.placeholder(tf.float32, [None, ], name="actions_value")
			self.W = [None]*len(hidden_units)
			self.b = [None]*len(hidden_units)
			with tf.name_scope('parameters' + self.scope):
				for i in range(len(hidden_units)):
					if i == 0: 
						prev_size = self.n_obs
					else:
						prev_size = hidden_units[i-1]
					next_size = hidden_units[i]
					self.W[i] = tf.get_variable('W' + str(i), [next_size, prev_size], initializer = tf.contrib.layers.xavier_initializer(seed=1))
					self.b[i] = tf.get_variable('b' + str(i), [next_size, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
			
			self.W_out = tf.get_variable('W_out', [self.n_act, hidden_units[-1]], initializer=tf.contrib.layers.xavier_initializer(seed=1))
			self.b_out = tf.get_variable('b_out', [self.n_act, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
			
			self.carry = self.X
			with tf.name_scope('h1' + self.scope):
				for i in range(len(hidden_units)):
					self.carry = tf.add(tf.matmul(self.W[i], self.carry), self.b[i])
					self.carry = tf.nn.relu(self.carry)

			logits = tf.add(tf.matmul(self.W_out, self.carry), self.b_out)
			logits = tf.transpose(logits)
			self.action_probs  = tf.nn.softmax(logits, name='action_probs')
			with tf.name_scope('loss' + self.scope):
				self.log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=tf.transpose(self.Y))
				self.loss     = tf.reduce_mean(self.log_prob*self.training_rewards)

			with tf.name_scope('train' + self.scope):
				self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

		elif network == 'conv':
			raise NotImplementedError
	def choose_action(self, observation):
		observation  = observation[:, None]
		action_probs = self.sess.run(self.action_probs, feed_dict={self.X:observation})
		action       = np.random.choice(range(self.n_act), p=action_probs.ravel())
		return(action)

	def discount_rewards(self):
		running_mean = 0
		discounted_rewards = np.zeros(len(self.episode_rewards))
		for i in reversed(range(len(self.episode_rewards))):
			running_mean = running_mean * self.gamma + self.episode_rewards[i]
			discounted_rewards[i] = running_mean

		discounted_rewards -= np.mean(discounted_rewards)
		discounted_rewards /= np.std(discounted_rewards)
		return(discounted_rewards)

	def learn(self):
		input_feed = np.vstack(self.episode_obs).T
		action_feed = np.vstack(self.episode_actions).T
		reward_feed = self.discount_rewards()
		self.sess.run(self.train_op, feed_dict = {
			self.X:input_feed,
			self.Y:action_feed,
			self.training_rewards:reward_feed})
		self.episode_obs, self.episode_actions, self.episode_rewards = [], [], []

		return(reward_feed)

	def store_transition(self, o, a, r):
		self.episode_obs.append(o)
		self.episode_rewards.append(r)
		action_onehot = np.zeros(self.n_act)
		action_onehot[a] = 1
		self.episode_actions.append(action_onehot)


class GaussianNetwork(object):
	def __init__(self, sess, layers, state_dim, pred_dim, learning_rate, gamma, batch_size, com_net=None):
		self.sess       = sess
		self.s_dim      = state_dim
		self.pred_dim   = pred_dim
		self.lr         = learning_rate
		self.batch_size = batch_size
		self.gamma      = gamma
		self.com_net    = com_net
		self.episode_rewards = []
		self.episode_obs     = []
		self.episode_actions = []

		self.mu_inputs, self.mu, self.final_features = self.create_network(layers)
		#self.sigma_inputs, self.log_sigmas, _ = self.create_network(sigma_layers)
		#self.sigma = tf.log(tf.add(tf.exp(log_stds), tf.ones_like(log_stds)))
		self.sigma = tf.placeholder(tf.float32, [None, 1])
		self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
		self.action = self.normal_dist._sample_n(1)
		self.action_sy = tf.placeholder(tf.float32, [None, self.pred_dim])

		self.targets  = tf.placeholder(tf.float32, [None, 1])
		self.baseline = tf.placeholder(tf.float32, [None, 1])

		
		self.log_probs = -1./2 * self.normal_dist.log_prob(self.action_sy)

		#self.loss = self.log_probs * (self.targets - self.baseline)
		self.loss = self.log_probs * self.targets
		self.loss -= 1e-6 * self.normal_dist.entropy()
		self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


	def create_network(self, layers):
		inputs = tflearn.input_data(shape=[None, self.s_dim])
		net    = inputs
		for dim in layers:
			net = tflearn.fully_connected(net, dim)
			net = tflearn.layers.normalization.batch_normalization(net)
			net = tflearn.activations.relu(net)        

		w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
		out = tflearn.fully_connected(
			net, self.pred_dim, activation='linear', weights_init=w_init)

		return inputs, out, net

	def train(self, inputs, targets):
		self.sess.run(self.optimize, feed_dict={
			self.mu_inputs: inputs,
			self.targets: targets
		})

	def predict(self, inputs):
		sigma_feed = 1e-3*np.ones((inputs.shape[0], 1))
		a_pred = self.sess.run(self.action, feed_dict={
			self.mu_inputs: inputs,
			self.sigma:sigma_feed
			})
		if self.com_net is not None:
			com = self.com_net.predict(inputs.reshape(-1, self.s_dim))
		return com.reshape(-1,)[:2] + a_pred.reshape(-1,)

	def choose_action(self, inputs):
		return self.predict(inputs)

	def eval_loss(self, inputs, targets):
		return self.sess.run(self.loss, feed_dict={
			self.inputs:inputs,
			self.targets:targets
		})

	def discount_rewards(self):
		running_mean = 0
		discounted_rewards = np.zeros(len(self.episode_rewards))
		for i in reversed(range(len(self.episode_rewards))):
			running_mean = running_mean * self.gamma + self.episode_rewards[i]
			discounted_rewards[i] = running_mean

		discounted_rewards -= np.mean(discounted_rewards)
		if np.std(self.episode_rewards) > 1:
			discounted_rewards /= np.std(discounted_rewards)
		return(discounted_rewards)

	def learn(self):
		import IPython as ipy
		#ipy.embed()
		input_feed  = np.vstack(self.episode_obs)
		action_feed = np.vstack(self.episode_actions)
		reward_feed = self.discount_rewards().reshape(-1,1)
		sigma_feed  = .01 * np.ones((len(self.episode_obs), 1))
		self.sess.run(self.optimize, feed_dict = {
			self.mu_inputs:input_feed,
			self.action_sy:action_feed,
			self.targets:reward_feed,
			self.sigma:sigma_feed})

		self.episode_obs, self.episode_actions, self.episode_rewards = [], [], []

		return(reward_feed)

	def store_transition(self, o, a, r):
		self.episode_obs.append(o)
		self.episode_rewards.append(r)
		self.episode_actions.append(a)


class MLPNetwork(object):
	def __init__(self, sess, layers, state_dim, pred_dim, learning_rate, batch_size):
		self.sess       = sess
		self.s_dim      = state_dim
		self.pred_dim   = pred_dim
		self.lr         = learning_rate
		self.batch_size = batch_size

		self.inputs, self.out, self.final_features = self.create_network(layers)
		self.targets = tf.placeholder(tf.float32, [None, pred_dim])

		self.network_params = tf.trainable_variables()

		self.loss     = tflearn.mean_square(self.out, self.targets)
		self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


	def create_network(self, layers):
		inputs = tflearn.input_data(shape=[None, self.s_dim])
		net    = inputs
		for dim in layers:
			net = tflearn.fully_connected(net, dim)
			net = tflearn.layers.normalization.batch_normalization(net)
			net = tflearn.activations.relu(net)        

		w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
		out = tflearn.fully_connected(
			net, self.pred_dim, activation='linear', weights_init=w_init)

		return inputs, out, net

	def train(self, inputs, targets):
		self.sess.run(self.optimize, feed_dict={
			self.inputs: inputs,
			self.targets: targets
		})

	def predict(self, inputs):
		return self.sess.run(self.out, feed_dict={
			self.inputs: inputs
		})

	def extract_features(self, inputs):
		return self.sess.run(self.final_features, feed_dict={
			self.inputs: inputs
		})

	def eval_loss(self, inputs, targets):
		return self.sess.run(self.loss, feed_dict={
			self.inputs:inputs,
			self.targets:targets
		})


class ConvNet(nn.Module):
	def __init__(self, conv_layers, fc_layers):
		super().__init__()
		self.num_conv_layers = len(conv_layers)-1
		self.num_fc_layers   = len(fc_layers)-1
		
		self.maxpool_kernel  = 4
		self.conv_activation = lambda x: F.relu(F.max_pool2d(x, self.maxpool_kernel))
		self.fc_activation   = lambda x: F.elu(x)

		self.conv = nn.ModuleList([nn.Conv2d(conv_layers[i], conv_layers[i+1], kernel_size=5) for i in range(self.num_conv_layers)])
		self.fc = nn.ModuleList([nn.Linear(fc_layers[i],fc_layers[i+1]) for i in range(self.num_fc_layers)])

		for w in self.conv:
			w.weight.data.normal_(0, 1e-3)
			w.bias.data.normal_(0,1e-5)

		for w in self.fc:
			w.weight.data.normal_(0, 1e-3)
			w.bias.data.normal_(0,1e-5)


	def forward(self, x):
		for ic in range(self.num_conv_layers):
			x = self.conv_activation(self.conv[ic](x))
		num_feat = functools.reduce(operator.mul, x.size()[1:], 1)
		x = x.view(-1, num_feat)
		for i in range(self.num_fc_layers-1):
			x = F.relu(self.fc[i](x))
		x = self.fc[-1](x)
		return x

	@property
	def num_parameters(self):
		num_p = 0
		for p in self.parameters():
			num_p += functools.reduce(operator.mul, p.size(), 1)
		return(num_p)


class FcNet(nn.Module):
	def __init__(self, layers):
		super().__init__()
		self.num_layers = len(layers)-1
		self.fc = nn.ModuleList([nn.Linear(layers[i],layers[i+1]) for i in range(self.num_layers)])
		self.state_dim = layers[0]

	def forward(self, x):
		x = x.view(-1, self.state_dim)
		for i in range(self.num_layers-1):
			x = F.relu(self.fc[i](x))
		x = self.fc[-1](x)
		return x
	@property
	def num_parameters(self):
		num_p = 0
		for p in self.parameters():
			num_p += functools.reduce(operator.mul, p.size(), 1)
		return(num_p)


class GaussianPolicyTorch(nn.Module):
	def __init__(self, layers):
		super().__init__()
		self.num_layers = len(layers) - 1
		self.fc         = nn.ModuleList([nn.Linear(layers[i],layers[i+1]) for i in range(self.num_layers)])
		self.state_dim  = layers[0]
		self.pred_dim   = layers[-1]
		for w in self.fc:
			w.weight.data.normal_(0, 1e-3)
			w.bias.data.normal_(0,1e-5)

	def forward(self, x):
		x = x.view(-1, self.state_dim)
		for i in range(self.num_layers-1):
			x = F.relu(self.fc[i](x))
		x = self.fc[-1](x)
		return x
	def action_mu(self, x):
		return self.forward(x)

	def action_sigma(self, x):
		return 1e-3*torch.ones(self.pred_dim)



class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()
        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.logstd = AddBias(torch.zeros(num_outputs))
        self.fc_mean.weight.data.normal_(0,1e-4)
        self.fc_mean.bias.data.normal_(0, 1e-4)
        self.std_scale = 1e-4


    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = Variable(torch.zeros(action_mean.size()), volatile=x.volatile)
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return action_mean, action_logstd

    def sample(self, x, deterministic):
        action_mean, action_logstd = self(x)
        action_logstd = action_logstd
        action_std = action_logstd.exp()
        action_std = action_std * self.std_scale

        if deterministic is False:
            noise = Variable(torch.randn(action_std.size()))
            if action_std.is_cuda:
                noise = noise.cuda()
            action = action_mean + action_std * noise
        else:
            action = action_mean
        return action

    def logprobs_and_entropy(self, x, actions):
        action_mean, action_logstd = self(x)

        action_std = action_logstd.exp() * self.std_scale

        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(-1, keepdim=True)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + action_logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return action_log_probs, dist_entropy






class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))
        self._bias.data.normal_(0,1e-5)
    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias











