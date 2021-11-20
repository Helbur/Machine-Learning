import numpy as np
import matplotlib.pyplot as plt
import keras
np.random.seed(1)

class FFNN:
    def __init__(self, w, X):
        self.w = w #array of weights
        self.X = X #input vector
        
    def MSE(self, out, actual):
        return np.sum(np.sqrt((out - actual)**2))/np.len(actual)
    
    def crossEnt(out, actual):
        pass
    
    def sigmoid(self, x):
        return 1./(1+np.exp(-x))
    
    def sigmoid_deriv(self, x):
        return -np.exp(-x)/(1+np.exp(-x))**2
    
    def ReLU(self, x):
        return x * (x>0)
    
    def ReLU_deriv(self, x):
        return x>0
    
    def feedforward(self):
        i = 0
        self.z = []
        temp = self.X
        while i < len(self.w)-1:
            self.z.append(self.w[i].dot(temp))
            temp = self.sigmoid(self.z[-1])
            i += 1
        temp = self.w[-1].dot(temp)
        return temp
    
    def backprop(self, out):
        deltas = []
        # 1
        deltas.append(self.sigmoid_deriv(out))
        # 2
        n = len(self.w)
        for i in range(1,n):
            deltas.append(deltas[n-i].dot(self.w[n-i].T).dot(self.sigmoid_deriv(self.z[n-i-1])))
        # 3
        dEdb = deltas
        dEdw = [deltas[l]*self.sigmoid(self.z[l-1]) for l in range(1,n)]
        return dEdb, dEdw
    
    def SGDM(self, eta):
        #Momentum
        dw = 
        w = w - eta*self.backprop(sample) + eta*dw
        return w
    
    def train(self, iters):
        for i in range(iters):
            

# Setup linear regression model
N = 1000
noise = np.random.normal(0.0, 0.1, N)
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# Preprocess train, test, validation sets
x = np.random.uniform(-1,1,N)
y = x
z = FrankeFunction(x,y) + noise # function to be learned
X = np.vstack((np.ones_like(x), x)).T
Xtrain = X[:int(0.9*N)]; ztrain = z[:int(0.9*N)]
Xtest = X[int(0.9*N):]; ztest = z[int(0.9*N):]

# Train network
iterations = 10
wStruct = (2,4,2)
weights = [] #holds weight matrices
for i in range(len(wStruct)-1):
    weights.append(np.random.normal(0.0, 1.0,size=(wStruct[i],wStruct[i+1])))

NeurNet = FFNN(weights, Xtrain)
for i in range(iterations):
    NeurNet.feedforward()
    NeurNet.SGDM()