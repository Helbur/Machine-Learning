import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso

np.random.seed(42)

class FFNN:
    def __init__(self, hlayers, inp, target, eta=0.1):
        """ Constructs a feedforward neural network where the hlayer parameter
            accepts an integer array of any desired length containing the sizes
            of each hidden layer. Returns randomly initialized weight matrices"""
        self.target = target    # true output
        self.inp = inp       # input
        self.no_inputs = np.matrix(self.inp).shape[1]
        self.features = np.matrix(self.inp).shape[0]
        self.eta = eta       # learning rate
        self.l = len(hlayers) + 2  # depth of network
        # creates room for input, output layers
        self.layers = hlayers
        self.layers.insert(0,0); self.layers.insert(len(self.layers),0)
        self.layers = np.array(hlayers)
        self.set_weights_biases()
    
    # def set_layer_structure(self):
    #     """ To be used  """
    #     self.layers[0] = np.shape(self.inp)[0]
    #     self.layers[-1] = np.shape(self.target)[0]
    
    def set_weights_biases(self):
        """ It's useful to keep this subroutine separate from the
            rest of the initialization procedure due to minibatches """
        self.layers[0] = self.features
        self.layers[-1] = np.matrix(self.target).shape[0]
        self.wts = []   # weight matrices
        self.activations = [self.inp] # keeps track of activation function firings for use during backprop
        for i in range(self.l-1):
            # Initializes normally distributed weight matrices
            self.wts.append(np.random.randn(self.layers[i], self.layers[i+1]))
        self.bias = []
        for i in range(1,self.l):
            # Initializes biases
            self.bias.append(np.zeros_like(self.layers[i]) + 0.01)
    
    def sigmoid(self, x):
        """ The activation function, in this case logistic """
        return 1./(1+np.exp(-x))
    
    def sigmoidDeriv(self, x):
        return 0.5/(1+np.cosh(x))
    
    def ReLU(self, x):
        return np.max(np.zeros_like(x), x, axis=0)
    
    def MSE(self, outp):
        return np.mean((self.target - outp)**2)

    def GD(self):
        """ Ordinary gradient descent, advances 1 step """
        self.wts[0] -= self.eta*self.activations[0] @ self.deltas[0]
        for i in range(1,self.l-1):
            self.wts[i] -= self.eta*self.activations[i] @ self.deltas[i]
            self.bias[i] -= self.eta*self.deltas[i]
    
    def feedforward(self, inp):
        """ Passes the input through the network once """
        for i in range(self.l-1):
            old = self.activations[i] @ self.wts[i]
            new = self.sigmoid(old)
            self.activations.append(new)
        return self.activations[-1]
    
    def feedforward_out(self, X):
        for i in 
    
    def backprop(self):
        """ Standard backpropagation algorithm
            1. Activate input layer
            2. Feedforward
            3. Calculate output error
            4. Propagate backwards, calculate errors at each neuron
            5. Calculate gradient """
        outp = self.feedforward(self.inp)
        outpErr = outp - self.target
        # Obtain hidden errors recursively:
        deltas = [outpErr*self.sigmoidDeriv(outp)]
        for i in reversed(range(0,self.l-1)):
            deltas.append(deltas[-1] @ self.wts[i].T * \
                         self.sigmoidDeriv(self.activations[i]))
        self.deltas = deltas[::-1]
    
    def train(self, epochs=10, batch_size=100):
        """ Trains network using SGD by default, assuming train/test splitting
            has been performed"""
        for i in range(epochs):
            for j in range(len(self.inp) // batch_size):
                datapts = np.random.choice(
                    np.arange(len(self.inp)), size=batch_size, \
                        replace=False)
                self.inp = self.inp[datapts]
                self.target = self.target[datapts]
                self.set_weights_biases()
                self.feedforward(self.inp)
                self.backprop()
    
    def test(self, X_test, y_test):
        """ Tests model on novel data """
        pass

# Example data for testing the network
n = 2000 #datapoints
x = np.linspace(-1,1,n) #input data
sigma = 0.5 #noise factor
y = 2*x + sigma*np.random.randn(n)#generated data with n.dist. noise
Net = FFNN([50], x, y)
Net.train()