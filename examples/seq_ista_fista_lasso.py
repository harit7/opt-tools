# Sequential ISTA
import numpy as np
from numpy.linalg import norm
from math import sqrt

import time

import sys

COMMONS_PATH = "../commons/"
sys.path.append(COMMONS_PATH)

import utils
from ista_core import ista
from fista_core import fista


#Evaluates the least square function.
def fun_least_squares_seq(x, A, b):
    
    n,d  = A.shape
    x    = x.reshape(1, d)
    loss = (A.dot(x.T) - b) ** 2
    return (np.sum(loss, axis=0) / (2. * n))[0]

#Evaluates the gradient of the least square function.
def grad_fun_least_squares(x, AtA, Atb):
    
    n,d = A.shape
    x   = x.reshape(1, d) 

    g = (AtA.dot(x.T) - Atb)/n
    
    return g.T

np.random.seed(1)
X   = np.loadtxt("../../data/synthetic_reg_lin_1000_2.csv",delimiter=",")

n,m = X.shape
d   = m-1


A   = X[:,range(d)]
b   = X[:,range(d,d+1)]

AtA = np.dot(A.T,A)
Atb = np.dot(A.T,b)

x_0 = np.random.rand(d)


max_iters = 30
lamda     = 0.001

# f and gradient

f      = lambda x: fun_least_squares_seq(x, A, b)
grad_f = lambda x: grad_fun_least_squares(x, AtA, Atb)

# g, F and prox.
g      = lambda x: lamda * np.abs(x).sum()
F      = lambda x: f(x) + g(x)
prox_g = lambda x, l: utils.soft_threshold(lamda,x)

step   = norm(A.T.dot(A) /n, 2)

#ista_inspector = inspector(loss_fun=F, x_real=params, verbose=True)
x_star,obj_arr,grad_arr = ista(x_0,obj_fun=F, grad_f=grad_f, prox=prox_g, max_iters=max_iters, step=step, abs_tol=0.001)
print "x_star = "+ str(x_star.reshape(d))
