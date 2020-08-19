import numpy as np
def ReLU(x):

    return np.maximum(x,0)


def dReLU(x,y):
     x[y == 0]=0
     return x