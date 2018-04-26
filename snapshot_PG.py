import tensorflow as tf
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.distributions import Categorical



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




class FcNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.num_layers = len(layers)-1
        self.fc = nn.ModuleList([nn.Linear(layers[i],layers[i+1]) for i in range(self.num_layers)])
        for layer in self.fc:
            torch.nn.init.xavier_uniform(layer.weight)

    def forward(self, x):
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













