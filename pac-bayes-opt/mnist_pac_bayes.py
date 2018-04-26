import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
import math

from mnist_data import get_train_data, get_test_data
from models import FcNet, FcNetStochastic
from models import train_model, test_model
from torch.autograd import Variable
import torch.nn as nn

import itertools
from utils import *

def get_batch(train_loader):
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = Variable(data), Variable(target)
		break
	return(data, target)


train_loader = get_train_data(32)

test_loader  = get_test_data(10000)
for batch_idx, (data, target) in enumerate(test_loader):
	test_data, test_target = Variable(data), Variable(target)

layers = [784, 200, 200, 2]
model  = FcNetStochastic(layers, 0)
model(get_batch(train_loader)[0])


lr           = .01
momentum     = .5
log_interval = 1000
epochs       = 20
fc           = True
binary       = True

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)



nll_coef=1
logistic_coef=0
for epoch in range(6):
	train_model(model, optimizer, train_loader, epoch, binary, nll_coef, logistic_coef)
	test_model(model, test_loader, binary, nll_coef, logistic_coef)


m = 60000
train_loader = get_train_data(m)
for batch_idx, (data, target) in enumerate(train_loader):
	train_data, train_target = Variable(data), Variable(target)
	if binary==True:
		train_target[target<5]  = 0
		train_target[target>=5] = 1


# values in paper: 
# b = 100, c = .1, delta = .025
b = 100
c = .1
delta = .025

d = model.num_parameters

zeta = Variable(torch.abs(flatten_model_parameters(model)[0]) - 2,requires_grad=True) 
rho     = Variable(torch.FloatTensor([-3]), requires_grad=True)
s       = torch.exp(2*zeta)
lambda_ = torch.exp(2*rho)




model_stochastic = FcNetStochastic(layers, s)
model_stochastic.load_state_dict(copy.deepcopy(model.state_dict()))
#test_model(model_stochastic, test_loader, binary, nll_coef, logistic_coef)

model_stochastic.train() # enables dropout
#torch_B_RE(zeta, s, zeta, lambda_, delta, b, c, m)

optimizer = optim.Adam(itertools.chain(model_stochastic.parameters(), [rho, zeta]), lr=.0001)


log_interval = 1
test_interval = 100

for epoch in range(10000):
	w      = flatten_model_parameters(model_stochastic)[0]
	model_stochastic.noise_scale = s
	optimizer.zero_grad()
	output = model_stochastic(train_data)
	w0     = w

	#s = lambda_*Variable(torch.ones(d))
	loss0 = F.nll_loss(output, train_target)

	loss1 = 1/lambda_ * torch.norm(s, p=1) + 1/lambda_ * torch.norm(w-w,p=2)**2 + d * torch.log(lambda_)
	loss1 -= (torch.sum(torch.log(s)) + d)
	loss1 /= 2*2*(m-1)	
	loss2 = 2 * torch.log(b * torch.log(c/lambda_)) / (2*(m-1))
	loss3 = Variable((torch.log(torch.from_numpy(np.array([math.pi**2 * m / (6*delta)]))) / (2*(m-1))).float(), requires_grad=False)

	loss = loss0+loss1+loss2+loss3
	loss.backward(retain_graph=True)
	optimizer.step()
	s       = torch.exp(2*zeta)
	lambda_ = torch.exp(2*rho)


	if epoch % log_interval == 0:
		print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx * len(train_data), len(train_loader.dataset),
			100. * batch_idx / len(train_loader), loss.data[0]))
	if epoch % test_interval == 0:
		model_stochastic.noise_scale = 0
		test_model(model_stochastic, test_loader, binary, nll_coef, logistic_coef)


test_model(model_stochastic, test_loader, binary, nll_coef, logistic_coef)

















x, _ = get_batch(train_loader)
model.sigma = .01
preds = []
for _ in range(100):
	preds.append(model(x).data.numpy()[:,1])

preds = np.vstack(preds)
var_est = np.mean(np.std(preds,axis=0))


optimizer = optim.Adam([s], lr=.0001)
#optimizer = optim.SGD(model_stochastic.parameters(), lr=1, momentum=momentum)

s  = Variable(torch.randn(1000), requires_grad=True)
s0 = Variable(torch.zeros(1000), requires_grad=False)
optimizer = optim.Adam([s], lr=.0001)

criterion = nn.MSELoss()

for i in range(150000):
	loss = criterion(s,s0)
	#optimizer.zero_grad()
	loss.backward()
	optimizer.step()



s         = Variable(torch.randn(1000), requires_grad=True)
optimizer = optim.Adam([s], lr=.01)

criterion = lambda x: torch.sum(torch.pow(x,2))

for i in range(150000):
	loss = criterion(s)
	#optimizer.zero_grad()
	loss.backward()
	optimizer.step()









