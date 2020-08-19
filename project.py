import numpy as np
from deeplearning.CNN import CNN_layer
from deeplearning.Pooling import Pooling
from deeplearning.Full_connection import Full_connection
from deeplearning.LRN import LRN
from deeplearning.Sigmoid import sigmoid,d_sigmoid,score,loss_function
from deeplearning.ReLU import ReLU,dReLU
import pickle
from matplotlib import pyplot as plt

file=open('../cifar-10-batches-py/data_batch_1','rb')
data=pickle.load(file,encoding='bytes')
s=data[b'data']
x=np.zeros((1,10000,3,32,32))
for i in range(10000):
    x[0,i,:,:,:]=s[i,:].reshape(3,32,32)
x=x[:,0:40,:,:,:]
p=data[b'labels']
y=np.zeros((10,20,1))
for i in range(20):
    r=int(p[i])
    y[r,i,:]=1

cnn1=CNN_layer(1,5,10)
pool=Pooling(3,1)
lrn=LRN()
cnn2=CNN_layer(1,5,25)
cnn3=CNN_layer(1,3,20)
cnn4=CNN_layer(1,3,20)
fn1=Full_connection(25)
fn2=Full_connection(25)
pool1=Pooling(3,1)
pool2=Pooling(3,1)
pool3=Pooling(3,1)
pool4=Pooling(3,1)
softmax=Full_connection(10)
number_list=[]
scores=[]
for number in range(25):
 for a in range(19):
    c1=cnn1.fit(x[:,a,:,:,:])
    l1=lrn.fit_another(c1)
    r1=ReLU(l1)
    r1=pool1.fit_mean(r1)
    c2 = cnn2.fit(r1)
    l2 = lrn.fit_another(c2)
    r2= ReLU(l2)
    r2 = pool2.fit_mean(r2)
    c3 = cnn3.fit(r2)
    l3= lrn.fit_another(c3)
    r3 = ReLU(l3)
    r3 = pool3.fit_mean(r3)
    c4 = cnn4.fit(r3)
    l4 = lrn.fit_another(c4)
    r4 = ReLU(l4)
    r4 = pool4.fit_mean(r4)
    r_r=r4.reshape((-1,1))
    f1=fn1.fit(r_r)
    f2=fn2.fit(ReLU(f1))
    s_=softmax.fit(ReLU(f2))
    result=sigmoid(s_)
    score_=score(result,y[:,a,:])
    loss_=loss_function(result, y[:,a,:])
    dx_so=d_sigmoid(loss_)
    dx_f2_be=softmax.g_fc(dx_so)
    softmax.momentum()
    dx_f2=fn2.g_fc(dReLU(dx_f2_be,f2))
    fn2.momentum()
    dx_f1=fn1.g_fc(dReLU(dx_f2,f1))
    fn1.momentum()
    dcn5 = pool4.g_mean(dx_f1.reshape((r4.shape)))
    dcn4 = cnn4.g_cnn(lrn.g_another(dReLU(dcn5,l4)))
    dcn4 = pool3.g_mean(dcn4)
    cnn4.momentum()
    dcn3 = cnn3.g_cnn(lrn.g_another(dReLU(dcn4,l3)))
    dcn3 = pool2.g_mean(dcn3)
    cnn3.momentum()
    dcn2 = cnn2.g_cnn(lrn.g_another(dReLU(dcn3,l2)))
    cnn2.momentum()
    dcn2 = pool1.g_mean(dcn2)
    dcn1 = cnn1.g_cnn(lrn.g_another(dReLU(dcn2,l1)))
    cnn1.momentum()
 number_list.append(number)
 c1=cnn1.fit(x[:,19,:,:,:])
 l1=lrn.fit(c1)
 r1=ReLU(l1)
 r1=pool1.fit_mean(r1)
 c2 = cnn2.fit(r1)
 l2 = lrn.fit(c2)
 r2= ReLU(l2)
 r2 = pool2.fit_mean(r2)
 c3 = cnn3.fit(r2)
 l3= lrn.fit(c3)
 r3 = ReLU(l3)
 r3 = pool3.fit_mean(r3)
 c4 = cnn4.fit(r3)
 l4 = lrn.fit(c4)
 r4 = ReLU(l4)
 r4 = pool4.fit_mean(r4)
 r_r=r4.reshape((-1,1))
 f1=fn1.fit(ReLU(r_r))
 f2=fn2.fit(ReLU(f1))
 s_=softmax.fit(ReLU(f2))
 result=sigmoid(s_)
 score_=score(result,y[:,19,:])
 scores.append(score_)
plt.scatter(number_list,scores)
plt.show()
d={'c1.theta':cnn1.cores,'c1.b':cnn1.b,
'c2.theta':cnn2.cores,'c2.b':cnn2.b,
'c3.theta':cnn3.cores,'c3.b':cnn3.b,
'c4.theta':cnn4.cores,'c4.b':cnn4.b,
'f1.theta':fn1.theta,'f1.b':fn1.b,
'f2.theta':fn2.theta,'f2.b':fn2.b,
'sm.theta':softmax.theta,'sm.b':softmax.b}
np.save('data_.npy',d)#每次要更改文件，读取时要更改self.n防止初始化