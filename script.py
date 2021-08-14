import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf


def kl_divergence(q,p,A,B,C,D,x,var_q):
	def formula(num):
		return q(num,x,C,D,var_q)*np.log(norm(num,0,1)/q(num,x,C,D,var_q)) #needs a variance for q

	return -1*quad(formula,-10,10)[0]

def E_q(p,A,B,C,D,var_p,var_q,x):
	def formula(num):
		return np.log(p(x,num,A,B,var_p))*q(num,x,C,D,var_q)

	return -1*quad(formula,-10,10)[0];		

def p_dist(x,z,A,B,var):
	return norm.pdf(x,A*z+B,var)

def q_dist(z,x,C,D,var):
	return norm.pdf(z,C*x+D,var)

def L(A,B,C,D,x,var_p,var_q):
	return E_q(p_dist,A,B,C,D,var_p,var_q,x) - kl_divergence(q_dist,p_dist,A,B,C,D,x)

if __name__ == '__main__':
	
	print('Enter the parameters for the truth.')
	print('B_0 =',end=' ')
	B_0 = float(input())
	print('A_0 =',end=' ')
	A_0 = float(input())
	print('var =',end=' ')
	var = float(input())

	print('Enter the random seed: ', end='')
	np.random.seed(int(input()))

	print('Generating sample')
	S=np.random.normal(B_0,A_0*A_0+var,500)

	hist_dat = plt.hist(S,50)
	plt.show()
