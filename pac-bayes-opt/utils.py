import torch
import numpy as np
import numpy.linalg as nlg


def flatten_model_parameters(model):
	parameters = []
	for p in model.parameters():
		parameters.append(p.data.numpy().flatten())

	return(torch.FloatTensor(np.hstack(parameters)), parameters)



def kl_gaussian_diag(w, s, w0, lambda_):
	d = w.shape[0]
	term1 = 1/lambda_ * nlg.norm(s, ord=1)
	term2 = 1/lambda_ * nlg.norm(w-w0,ord=2)**2
	term3 = d * np.log(lambda_)
	term4 = np.sum(np.log(s))
	term5 = d
	res   = term1 + term2 + term3 - term4 - term5
	return(.5 * res)


def torch_kl_gaussian_diag(w, s, w0, lambda_):
	d = w.shape[0]
	term1 = 1/lambda_ * torch.norm(s, p=1)
	term2 = 1/lambda_ * torch.norm(w-w0,p=2)**2
	term3 = d * torch.log(lambda_)
	term4 = torch.sum(torch.log(s))
	term5 = d
	res   = term1 + term2 + term3 - term4 - term5
	return(.5 * res)

def B_RE(w, s, w0, lambda_, delta, b, c, m):
	# the B_RE version of the relaxed PAC-Bayes bound
	# appearing in equation 15
	# w is weights that we are optimizing for
	# s is the standard deviation of the weights,
	# so the stochastic neural net is given by N(w,s)
	# w0 is the weights returned by SGD on the deterministic net
	# trained on MNIST
	# lambda_ is the continuous constant we are solving for
	# which will later be replaced by b * log(c/lambda)
	# delta is the confidence parameter at which we want the 
	# PAC-Bayes bound to hold
	# b, c are scaling parameters for the scale of search for lambda_
	# m is the number of samples we are using to train, I think
	term1 = kl_gaussian_diag(w,s,w0,lambda_)
	term2 = 2 * np.log(b * np.log(c/lambda_))
	term3 = np.log(np.pi**2 * m / (6*delta))
	res   = (term1 + term2 + term3)/(2*(m-1))
	return(res)

def torch_B_RE(w, s, w0, lambda_, delta, b, c, m):
	# the B_RE version of the relaxed PAC-Bayes bound
	# appearing in equation 15
	# w is weights that we are optimizing for
	# s is the standard deviation of the weights,
	# so the stochastic neural net is given by N(w,s)
	# w0 is the weights returned by SGD on the deterministic net
	# trained on MNIST
	# lambda_ is the continuous constant we are solving for
	# which will later be replaced by b * log(c/lambda)
	# delta is the confidence parameter at which we want the 
	# PAC-Bayes bound to hold
	# b, c are scaling parameters for the scale of search for lambda_
	# m is the number of samples we are using to train, I think
	term1 = torch_kl_gaussian_diag(w,s,w0,lambda_)
	term2 = 2 * torch.log(b * torch.log(c/lambda_))
	term3 = np.log(np.pi**2 * m / (6*delta))
	res   = (term1 + term2 + term3)/(2*(m-1))
	return(res)



def test():
	w = torch.randn(8)
	s = torch.ones(8)
	w0 = w
	lambda_ = torch.ones(1)

	torch_kl = torch_kl_gaussian_diag(w,s,w0,lambda_)
	assert(torch_kl[0] == 0)

	w = np.random.randn(8)
	s = np.ones(8)
	w0 = w
	lambda_ = 1
	np_kl = kl_gaussian_diag(w,s,w0,lambda_)
	delta = .95
	b = 5
	c = 10
	m = 100
	bound = B_RE(w,s,w0,lambda_,delta,b,c,m)
	assert(np_kl == 0)






