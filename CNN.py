import numpy as np
class CNN_layer():
    def __init__(self,strided_number=1,size_number=3,core_number=1):
        self.input_number = None
        self.input_layer=None
        self.output_number = core_number
        self.cores=None
        self.size_number=size_number
        self.strided = strided_number
        self.output =None
        self.cores_re=None
        self.b=np.random.random()
        self.db=None
        self.x=None
        self.dx=None
        self.dtheta=None
        self.velocity_b = 0
        self.velocity_theta = 0
        self.first_momentum_theta = 0
        self.second_momentum_theta = 0
        self.first_momentum_b = 0
        self.second_momentum_b = 0
        self.n=0
    def fit(self,input):#input是四维数组
        if self.n ==0:
            self.cores = np.random.random((self.output_number, input.shape[1], self.size_number, self.size_number))-0.5
            self.cores_re=np.zeros((self.output_number, input.shape[1], self.size_number, self.size_number))
        else:
            self.cores=self.cores
            self.cores_re=np.zeros((self.output_number, input.shape[1], self.size_number, self.size_number))
        self.x=input
        self.input_number = input.shape[0]
        self.input_layer=input.shape[1]
        ilength=input.shape[3]
        iwidth=input.shape[2]
        clength=self.size_number
        output = np.ones((input.shape[0],self.output_number, int((iwidth - clength) / self.strided + 1),
                          int((ilength - clength) / self.strided + 1)))
        for lc in range(output.shape[0]):# 对于每个样本
            for ic in range(output.shape[1]):  # 对于每一层
                for jc in range(output.shape[2]):  # 对于每一列
                    for kc in range(output.shape[3]):  # 对于每一行
                        cnn_result = np.sum(np.multiply(input[lc,:,
                        jc*self.strided:jc*self.strided+clength,kc*self.strided:kc*self.strided+clength],self.cores[ic,:,:,:]))
                        output[lc,ic,jc,kc]=cnn_result
        self.output=output+self.b
        self.n+=1
        return self.output

    def fit_dx(self, input, cores):  # input为五维度向量，一维batch,二维输出数，三维输入channel数，四维宽，五维长对于dx
        clength = self.size_number
        output = np.ones((self.x.shape))
        for lc in range(output.shape[0]):  # 对于每个样本
            for ic in range(output.shape[1]):  # 对于每一层
                for jc in range(output.shape[2]):  # 对于每一列
                    for kc in range(output.shape[3]):  # 对于每一行
                        cnn_result = np.sum(np.multiply(input[lc, :, :,
                                                        jc * self.strided:jc * self.strided + clength,
                                                        kc * self.strided:kc * self.strided + clength],
                                                        cores[:, :, :, :])) / self.x.shape[0]
                        output[lc, ic, jc, kc] = cnn_result
        output = output + self.b
        return output
    def fit_dtheta(self, input, cores):
        clength = cores.shape[-1]
        output = np.ones((self.cores.shape))
        sum=0
        for lc in range(output.shape[0]):  # 对于每个核
            for ic in range(output.shape[1]):  # 对于每一层
                for jc in range(output.shape[2]):  # 对于每一列
                    for kc in range(output.shape[3]):  # 对于每一行
                        for b in range(input.shape[0]):#对于每个样本
                            sum += np.sum(np.multiply(input[b, ic,
                                                        jc * self.strided:jc * self.strided + clength,
                                                        kc * self.strided:kc * self.strided + clength],
                                                        cores[b, lc, :, :])) / input.shape[0]
                        output[lc, ic, jc, kc] = sum
                        sum=0
        return output
    def g_cnn(self,input):
        output_rearrange=np.zeros((self.output.shape[0],self.output.shape[1],self.input_layer,self.output.shape[2]+self.x.shape[-1]-1,self.output.shape[3]+self.x.shape[-1]-1))
        for ic1 in range(input.shape[0]):
            for kc1 in range(input.shape[1]):
                for lc1 in range(output_rearrange.shape[2]):
                    output_rearrange[ic1,kc1,lc1,self.size_number-1:self.size_number-1+self.output.shape[3],
            self.size_number-1:self.size_number-1+self.output.shape[3]]=input[ic1,kc1,:,:]#此处输入与上函数输出大小一致
        for ic1 in range(self.cores.shape[0]):
            for kc1 in range(self.cores.shape[1]):
                self.cores_re[ic1,kc1,:,:]=np.flipud(np.fliplr(self.cores[ic1,kc1,:,:]))
        self.dx=self.fit_dx(output_rearrange,self.cores_re)
        cores_re1=np.zeros((input.shape))
        #for ic2 in range(input.shape[0]):
            #for kc2 in range(input.shape[1]):
                #cores_re1[ic2,kc2,:,:]=input[ic2,kc2,:,:]
        self.dtheta=self.fit_dtheta(self.x,input)
        self.db=np.sum(input)
        return self.dx

    def momentum(self,rho=0.9, alpha=0.0005):
        self.velocity_b=rho*self.velocity_b+(1-rho)*self.db
        self.b-=alpha*self.velocity_b
        self.velocity_theta = rho * self.velocity_theta + (1-rho)*self.dtheta
        self.cores -= alpha * self.velocity_theta
    def Adam(self,beta1,beta2,alpha):
        self.first_momentum_b = beta1 * self.first_momentum_b + (1 - beta1) * self.db
        self.second_momentum_b = beta2 * self.first_momentum_b + (1 - beta2) * self.db
        self.b+= alpha*self.first_momentum_b / (np.sqrt(self.second_momentum_b) + 1e-7)
        self.first_momentum_theta = beta1 * self.first_momentum_theta + (1 - beta1) * self.dtheta
        self.second_momentum_theta = beta2 * self.first_momentum_theta + (1 - beta2) * self.dtheta
        self.dtheta = self.first_momentum_theta / (np.sqrt(self.second_momentum_theta) + 1e-7)