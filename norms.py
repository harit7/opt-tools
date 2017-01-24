import numpy as np
from pav import pav

class OWLNorm:
    
    def __init__(self,c):
        assert all(c[i] >= c[i+1] for i in xrange(len(c)-1))
        self.c = c

        
    def compute(self,x):
        x_p =-np.sort(-1*np.abs(x)) #sorted in descending order of absolute values
        return np.dot(x_p,self.c)
    
    def prox(self,v,lmda):
    
        v_abs = np.abs(v)
        v_sort_idx = np.argsort(-v_abs) # argsort desc.
        v_sort_abs = v_abs[v_sort_idx] #sort desc.

        # project to the monotone cone.
        x   = - pav( - (v_sort_abs- lmda*self.c)) # pav is written for array in ascending order.

        # project to \mathbb{R}^n_+
        x   = x.clip(min=0) 

        y  =  np.zeros(len(x))

        y[v_sort_idx] = x
        y = np.multiply(y,np.sign(v))

        return y

class L1Norm:
        
    def compute(self,x):
        return np.abs(x).sum()
    
    def grad(self,x):
        return np.sign(x)
    
    def prox(self,v,lmda):
        return (v-lmda).clip(min=0) - (-v-lmda).clip(min=0)
    