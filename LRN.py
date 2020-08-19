import numpy as np
class LRN():
    def __init__(self):
        self.alpha=0.0001
        self.beta=0.75
        self.n=5
        self.k=2
        self.sum_result=0
        self.velocity = 0
        self.first_momentum = 0
        self.second_momentum = 0
        self.gv=None
    def fit(self,input):
        for m in range(input.shape[0]):
         for i in range(input.shape[1]):
            for k in range(input.shape[2]):
                for l in range(input.shape[3]):

                    if (i-(self.n)/2)>0 and i+self.n/2<input.shape[0]:
                        for p in range(int(i-self.n/2),int(i+self.n/2)):
                            self.sum_result+=input[m,p,k,l]

                    if i-self.n/2>0 and i+self.n/2>input.shape[0]:
                        for p in range(int(i-self.n/2), input.shape[0]):
                            self.sum_result += input[m,p, k, l]

                    if i-self.n/2<0 and i+self.n/2<input.shape[0]:
                        for p in range(0, int(i+self.n/2)):
                            self.sum_result += input[m,p, k, l]

                    input[m,i, k, l] = input[m,i, k, l] / (self.k + self.alpha * (self.sum_result) ** 2) ** self.beta
        return input
    def g(self,input):
        for m in range(input.shape[0]):
         for i in range(input.shape[1]):
            for k in range(input.shape[2]):
                for l in range(input.shape[2]):
                    if (i - (self.n) / 2) > 0 and i + self.n / 2 < input.shape[0]:
                        for p in range(int(i - self.n / 2), int(i + self.n / 2)):
                            self.sum_result += input[m,p, k, l]

                    if i - self.n / 2 > 0 and i + self.n / 2 > input.shape[0]:
                        for p in range(int(i - self.n / 2), input.shape[0]):
                            self.sum_result += input[m,p, k, l]

                    if i - self.n / 2 < 0 and i + self.n / 2 < input.shape[0]:
                        for p in range(0, int(i + self.n / 2)):
                            self.sum_result += input[m,p, k, l]
                    input[m,i,k,l]=1/(self.k+(self.sum_result)**2)**0.75-0.75*2*(input[m,i,k,l]**2)/(self.k+(self.sum_result)**2)**1.75
        self.gv=input
        return input

    def fit_another(self, input):
        for m in range(input.shape[0]):
            for i in range(input.shape[1]):
                for k in range(input.shape[2]):
                    for l in range(input.shape[3]):
                        self.sum_result=0
                        self.sum_result+=input[m,i,k,l]
                for k in range(input.shape[2]):
                    for l in range(input.shape[3]):
                        input[m,i,k,l]=input[m,i,k,l]/input[m,i, k, l] / (self.k + self.alpha * (self.sum_result) ** 2) ** self.beta
        return input
    def g_another(self,input):
        for m in range(input.shape[0]):
            for i in range(input.shape[1]):
                for k in range(input.shape[2]):
                    for l in range(input.shape[3]):
                        self.sum_result = 0
                        self.sum_result += input[m, i, k, l]
                for k in range(input.shape[2]):
                    for l in range(input.shape[3]):
                        input[m, i, k, l] = 1/(self.k+(self.sum_result)**2)**0.75-0.75*2*(input[m,i,k,l]**2)/(self.k+(self.sum_result)**2)**1.75
        return input