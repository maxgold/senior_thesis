import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import functools
import operator

batch_size = 32
test_batch_size = 1

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True)

def train(model, epoch, binary=False):
    model.train() # enables droput
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if binary==True:
            target[target<5]  = 0
            target[target>=5] = 1
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(model, binary=False):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if binary==True:
            target[target<5]  = 0
            target[target>=5] = 1
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train_regression(model, epoch, binary=False):
    model.train() # enables droput
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if binary==True:
            target[target<5]  = 0
            target[target>=5] = 1
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target.float())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test_regression(model, binary=False):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if binary==True:
            target[target<5]  = 0
            target[target>=5] = 1
        output = model(data)
        test_loss += F.mse_loss(output, target.float(), size_average=False).data[0] # sum up batch loss
        pred = np.argmin(np.abs(output.data.numpy()-np.arange(0,10)))
        correct += (pred == target.data.numpy()).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



class ConvNet(nn.Module):
    def __init__(self, conv_layers, fc_layers):
        super().__init__()
        self.num_conv_layers = len(conv_layers)-1
        self.num_fc_layers   = len(fc_layers)-1
        
        self.maxpool_kernel  = 2
        self.conv_activation = lambda x: F.relu(F.max_pool2d(x, self.maxpool_kernel))
        self.fc_activation   = lambda x: F.elu(x)

        self.conv = nn.ModuleList([nn.Conv2d(conv_layers[i], conv_layers[i+1], kernel_size=5) for i in range(self.num_conv_layers)])
        self.fc = nn.ModuleList([nn.Linear(fc_layers[i],fc_layers[i+1]) for i in range(self.num_fc_layers)])

    def forward(self, x):
        for ic in range(self.num_conv_layers):
            x = self.conv_activation(self.conv[ic](x))
        num_feat = functools.reduce(operator.mul, x.size()[1:], 1)
        x = x.view(-1, num_feat)
        for i in range(self.num_fc_layers-1):
            x = F.relu(self.fc[i](x))
        x = self.fc[-1](x)
        return F.log_softmax(x, dim=1)

    @property
    def num_parameters(self):
        num_p = 0
        for p in self.parameters():
            num_p += functools.reduce(operator.mul, p.size(), 1)
        return(num_p)


class ConvNetRegression(nn.Module):
    def __init__(self, conv_layers, fc_layers):
        super().__init__()
        self.num_conv_layers = len(conv_layers)-1
        self.num_fc_layers   = len(fc_layers)-1
        
        self.maxpool_kernel  = 2
        self.conv_activation = lambda x: F.relu(F.max_pool2d(x, self.maxpool_kernel))
        self.fc_activation   = lambda x: F.elu(x)

        self.conv = nn.ModuleList([nn.Conv2d(conv_layers[i], conv_layers[i+1], kernel_size=5) for i in range(self.num_conv_layers)])
        self.fc = nn.ModuleList([nn.Linear(fc_layers[i],fc_layers[i+1]) for i in range(self.num_fc_layers)])

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


class FcNetRegression(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.num_layers = len(layers)-1
        self.fc = nn.ModuleList([nn.Linear(layers[i],layers[i+1]) for i in range(self.num_layers)])

    def forward(self, x):
        x = x.view(-1, 784)
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


lr           = .01
momentum     = .5
log_interval = 1000
epochs       = 20
fc           = True
binary       = True


num_filter = 10
num_fc     = num_filter*16

model = ConvNet([1,5,num_filter],[num_fc,10,2])

model = FcNet([784, 10, 3, 2])

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


for epoch in range(1, epochs + 1):
    train(model, epoch, binary)
    test(model, binary)


num_filter = 20
num_fc     = num_filter*16

model = ConvNetRegression([1,10,num_filter],[num_fc,20, 10, 1])

#model = FcNet([784, 30, 30, 1])

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

epochs = 50


for epoch in range(1, epochs + 1):
    train_regression(model, epoch, binary)
    test_regression(model, binary)
















