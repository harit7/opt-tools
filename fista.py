# FISTA algorithm.
# The core structure of algorithm remains same for both parallel and sequential versions.
from numpy.linalg import norm
from math import sqrt
from scipy.optimize import line_search
import numpy as np

def fista(x_0,obj_fun, grad_f, prox, max_iters=1000, step=0.01, abs_tol=0.0001):
    print x_0.shape, type(x_0)
    x = x_0.copy()
    y = x_0.copy()
    h = 0.01
    obj_arr = [obj_fun(x)]
    grad_arr= [grad_f(x)]
    
    print "step = "+str(step)
    print "x_0 = "+str(x_0)
    beta = 0.1
    prev_step = step
    for i in range(1,max_iters):
        
        out = line_search(obj_fun,grad_f,x, -np.ones(len(x)))
        step = out[0]
        print step 
        if(step is None):
            step = prev_step
        prev_step = step
        
        g_i = grad_f(y)
        #print g_i.shape, type(g_i)
        x1  = prox(y - step * g_i, step)
        #print x1.shape, type(x1)

        h1  = (1.0 + sqrt(1.0 +4 *h*h ))/2

        y   = x1 + ((h-1)/(h1))*(x1 -x)

        x   = x1
        h   = h1

        obj = obj_fun(x)
        
        obj_arr.append(obj)
        grad_arr.append(g_i)
        #step = step*beta


        
        print "obj: "+str(obj) +"\t" "norm g_i: "+str( norm(g_i)) +"\t step : "+str(step)
        '''
        if( abs(obj_arr[i]-obj_arr[i-1]) <= abs_tol ):
            print "Converged in "+str(i) +" iterations"
            break
        '''            
    return x,obj_arr, grad_arr

