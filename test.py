import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from matplotlib import pyplot as plt


def kl_divergence(q,p,A,B,C,D):
	def formula(x):
		return q(x,C,D)*np.log(p(x,A,B)/q(x,C,D)) #try return zero if got weird value

	return -1*quad(formula,-np.inf,np.inf)[0]

def kl_alternate(q,p,A,B,C,D):
	def formula(x):
		if (p(x,A,B)==0 or q(x,C,D)==0):
			return 0
		else:
			return q(x,C,D)*np.log(p(x,A,B)/q(x,C,D))

	return -1*quad(formula,-np.inf,np.inf)[0]

def p(x,A,B):
	return norm.pdf(x,A,B)

def q(x,C,D):
	return norm.pdf(x,C,D)

if __name__ == '__main__':
	# print('%1.3f' %kl_divergence(q,p,0,2,2,2))
	# print('%1.3f' %kl_divergence(q,p,0,2,5,4))
	print('%1.3f' %kl_alternate(q,p,0,2,2,2))