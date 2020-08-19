import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
def d_sigmoid(x):
    return np.exp(-x)/(1+np.exp(-x))**2
def score(x,y):
    sum = 0
    for i in range(x.shape[0]):
        sum += np.exp(x[i, 0])
    return np.exp(x[y==1])/sum
def loss_function(x,y):
    sum = 0
    for i in range(x.shape[0]):
        sum += np.exp(x[i,0])
    for i in range(x.shape[0]):
        x[i,0]=np.exp(x[i,0])/sum-y[i,0]
    return x

