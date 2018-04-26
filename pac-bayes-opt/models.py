import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import copy
import functools
import operator

class FcNet(nn.Module):
	def __init__(self, layers):
		super().__init__()
		self.num_layers = len(layers)-1
		self.fc = nn.ModuleList([nn.Linear(layers[i],layers[i+1]) for i in range(self.num_layers)])

	def forward(self, x):
		x = x.view(-1, 784)
		for i in range(self.num_layers-1):
			x = F.relu(self.fc[i](x))
		x = self.fc[-1](x)
		return F.log_softmax(x, dim=1)
	@property
	def num_parameters(self):
		num_p = 0
		for p in self.parameters():
			num_p += functools.reduce(operator.mul, p.size(), 1)
		return(num_p)

class FcNetStochastic(nn.Module):
	def __init__(self, layers, noise_scale):
		super().__init__()
		self.num_layers = len(layers)-1
		self.fc = nn.ModuleList([nn.Linear(layers[i],layers[i+1]) for i in range(self.num_layers)])
		self.noise_scale  = noise_scale

	def forward(self, x):
		#import IPython as ipy
		#ipy.embed()
		self.set_noise(self.noise_scale)
		x = x.view(-1, 784)
		for i in range(self.num_layers-1):
			# this works because matrix multiplication is linear
			rand_weight = self.noise[i][0]
			rand_weight = rand_weight.view(self.fc[i].weight.shape)
			rand_weight = torch.transpose(rand_weight, 0, 1)
			if type(rand_weight) != Variable:
				rand_weight = Variable(rand_weight, requires_grad=False)
			rand_x      = torch.matmul(x, rand_weight)
			bias_term   = self.noise[i][1]
			if type(bias_term) != Variable:
				bias_term   = Variable(bias_term.view(self.fc[i].bias.shape), requires_grad=False)
			rand_x      = torch.add(rand_x, bias_term)
			x           = self.fc[i](x)
			x           = F.relu(x + rand_x)

		rand_weight = self.noise[-1][0]
		rand_weight = rand_weight.view(self.fc[-1].weight.shape)
		rand_weight = torch.transpose(rand_weight, 0, 1)
		if type(rand_weight) != Variable:
			rand_weight = Variable(rand_weight, requires_grad=False)
		rand_x = torch.matmul(x, rand_weight)
		bias_term = self.noise[-1][1]
		if type(bias_term) != Variable:
			bias_term = Variable(bias_term.view(self.fc[-1].bias.shape), requires_grad=False)
		rand_x = torch.add(rand_x, bias_term)
		x = self.fc[-1](x)
		x = F.relu(x + rand_x)
		return F.log_softmax(x, dim=1)
	def set_noise(self, noise_scale):
		if type(noise_scale) != Variable:
			noise = noise_scale * torch.randn(self.num_parameters)
		else:
			noise = noise_scale * Variable(torch.randn(self.num_parameters), requires_grad=False)
		self.noise = []
		counter = 0
		for i in range(self.num_layers):
			matrix_size = functools.reduce(operator.mul, self.fc[i].weight.size(), 1)
			bias_size   = self.fc[i].bias.size()[0]
			current = []
			current.append(noise[counter:counter+matrix_size])
			counter += matrix_size
			current.append(noise[counter:counter+bias_size])
			self.noise.append(current)

	@property
	def num_parameters(self):
		num_p = 0
		for p in self.parameters():
			num_p += functools.reduce(operator.mul, p.size(), 1)
		return(num_p)


def train_model(model, optimizer, train_loader, epoch, 
		binary=False, nll_coef=1, logistic_coef=0,
		log_interval=1000):
	model.train() # enables droput
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = Variable(data), Variable(target)
		if binary==True:
			target[target<5]  = 0
			target[target>=5] = 1
		optimizer.zero_grad()
		output = model(data)
		#import IPython as ipy
		#ipy.embed()
		loss1 = F.nll_loss(output, target)
		
		pos_probs = F.softmax(output,dim=1)[:,1]
		margin_target = binary_to_margin(target)
		margin_probs = binary_to_margin(pos_probs)
		loss2 = F.soft_margin_loss(margin_probs, margin_target)

		loss = nll_coef*loss1 + logistic_coef*loss2
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data[0]))

def test_model_fast(model, test_data, test_target, binary=False, nll_coef=1, logistic_coef=0):
	model.eval()
	test_loss = 0
	correct = 0
	if binary==True:
		test_target[test_target<5]  = 0
		test_target[test_target>=5] = 1
	output = model(test_data)
	loss1 = F.nll_loss(output, test_target, size_average=False).data[0]
	
	pos_probs = F.softmax(output,dim=1)[:,1]
	margin_target = binary_to_margin(test_target)
	margin_probs  = binary_to_margin(pos_probs)
	loss2 = F.soft_margin_loss(margin_probs, margin_target, size_average=False).data[0]

	test_loss += nll_coef*loss1 + logistic_coef*loss2
	pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
	correct += pred.eq(test_target.data.view_as(pred)).long().cpu().sum()

	test_loss /= len(test_data)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))


def test_model(model, test_loader, binary=False, nll_coef=1, logistic_coef=0):
	model.eval()
	test_loss = 0
	correct = 0
	for data, target in test_loader:
		data, target = Variable(data, volatile=True), Variable(target)
		if binary==True:
			target[target<5]  = 0
			target[target>=5] = 1
		output = model(data)
		loss1 = F.nll_loss(output, target, size_average=False).data[0]
		
		pos_probs = F.softmax(output,dim=1)[:,1]
		margin_target = binary_to_margin(target)
		margin_probs = binary_to_margin(pos_probs)
		loss2 = F.soft_margin_loss(margin_probs, margin_target, size_average=False).data[0]

		test_loss += nll_coef*loss1 + logistic_coef*loss2
		pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
		correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

	test_loss /= len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))


def binary_to_margin(dat):
	if type(dat.data) == torch.LongTensor:
		dat = long_to_floatVar(dat)
	dat = 2 * (dat - .5)
	return(dat)


def long_to_floatVar(dat):
	return(Variable(torch.FloatTensor(dat.data.numpy())))







