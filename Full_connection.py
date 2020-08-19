import numpy as np
class Full_connection():
    def __init__(self,output_number):
        self.number = 1
        self.theta=None
        self.b= np.random.random()
        self.output_number=output_number
        self.dtheta = None
        self.db = None
        self.x=None
        self.dx=None
        self.velocity_b=0
        self.velocity_theta = 0
        self.velocity_x = 0
        self.first_momentum_theta=0
        self.second_momentum_theta=0
        self.first_momentum_x = 0
        self.second_momentum_x = 0
        self.first_momentum_b = 0
        self.second_momentum_b = 0
        self.n=0
    def fit(self,X_train):#x,y为列向量
        self.x=X_train
        self.dx=np.zeros((X_train.shape))
        self.number = X_train.shape[0]
        if self.n == 0 :
            self.theta = np.random.random((self.number, self.output_number))-0.5
            self.dtheta = np.random.random((self.number, self.output_number))-0.5
        self.a=np.zeros((self.number,1))
        theta=self.theta
        for k in range(self.theta.shape[0]):
         for i in range(self.theta.shape[1]):
            if np.random.random()<0.5:
                self.theta[k,i]=0
        self.a=(self.theta.T).dot(X_train)+self.b
        self.theta=theta
        self.n+=1
        return self.a
    def g_fc(self,y):
        self.db=y
        for id1 in range(self.number):
            for id2 in range(self.output_number):
                self.dtheta=self.x.dot(y.T)
        for kd1 in range(self.number):
            for kd2 in range(y.shape[1]):
                self.dx=self.theta.dot(y)
        return self.dx
    def momentum(self,rho=0.9, alpha=0.0005):
        self.velocity_b=rho*self.velocity_b+(1-rho)*self.db
        self.b-=alpha*self.velocity_b
        self.velocity_theta=rho*self.velocity_theta+(1-rho)*self.dtheta
        self.theta-=alpha*self.velocity_theta
        return self
    def Adam(self,beta1,beta2,alpha):
        self.first_momentum_b = beta1 * self.first_momentum_b + (1 - beta1) * self.db
        self.second_momentum_b = beta2 * self.first_momentum_b + (1 - beta2) * self.db
        self.b += alpha*self.first_momentum_b / (np.sqrt(self.second_momentum_b) + 1e-7)
        self.first_momentum_theta = beta1 * self.first_momentum_theta + (1 - beta1) * self.dtheta
        self.second_momentum_theta = beta2 * self.first_momentum_theta + (1 - beta2) * self.dtheta
        self.theta += alpha*self.first_momentum_theta / (np.sqrt(self.second_momentum_theta) + 1e-7)
        return self
