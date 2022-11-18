import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
np.random.seed(1)

""" This is the code for stochastic gradient descent on OLS and Ridge
regression. The test polynomial is 2nd order. """

n = 1000    #datapoints
x = np.linspace(-3,3,n)
sigma = 0.5
y = 2 + 5*x - 3*x**2 + sigma*np.random.randn(n) #normally dist. noise
X = np.vstack((np.ones(n), x)).T # design matrix

def gradientOLS(X,y,beta):
    """ Gradient of the Mean Squared Error as function of weights """
    return X.T @ (X @ beta - y)

def gradientRidge(X,y,beta, lmbd):
    """ Gradient of Ridge cost function as function of weights and reg. parameter"""
    return gradientOLS(beta) + 2*lmbd*beta

def SGD(eta, k_batch, epochs):
    """ Stochastic Gradient Descent implemented with
        eta: learning rate
        k_batch: number of minibatches
        epochs: number of epochs """
    X_ = X
    np.random.shuffle(X_)
    betas = np.random.randn() # Normally dist weight initialization
    batch_size = X.shape[0] // k_batch
    for i in range(epochs):
        for j in range(k_batch):
            betas -= eta*gradientRidge(X[])