#importing libs

import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression


#Franke's function
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#Design matrix
def create_X(x, y, n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

#Fuction that calculates MSE
def Mean_Square_Error(y_predict, y_data):
    return np.mean((y_data-y_predict)**2)

#Function that calculates R2
def R2_Score_Function(y_predict, y_data):
    return (1 - np.sum((y_data - y_predict) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2))

# Make data.
N = 20  # spliting x, y into N poins
x = np.arange(0, 1, 1/N)
y = np.arange(0, 1, 1/N)
x, y = np.meshgrid(x,y)


#making noizy Fr function (note that x is (100,100), y is (100, 100), z is (10000,1) after reshape on 53 line
z = FrankeFunction(x, y)
z = z + np.random.normal(0, 0.05, size=z.shape)
z = z.reshape(-1,1)

#max polynomial degree
degree = 5

nlambdas = 500
lambdas = np.logspace(-3, 5, nlambdas)

MSEs = np.zeros(nlambdas)

i = 0
for lmd in lambdas:

    #creating X matrix on i polydegree
    X = create_X(x, y, degree)   
    #splitiing data
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)  
    
    I = np.zeros((X_train.shape[1], X_train.shape[1]))
    np.fill_diagonal(I, lmd)

    #calculating beta by matrix inversion
    beta = np.linalg.pinv(X_train.T @ (X_train) + I) @ (X_train.T) @ (z_train)  

    #test and train prediction 
    z_pred_test = X_test @ beta 
    #filling error and score arrays

    MSEs[i] = Mean_Square_Error(z_pred_test, z_test)
    i += 1


plt.figure()

plt.plot(np.log10(lambdas), MSEs, label = 'cross_val_score')

plt.xlabel('log10(lambda)')
plt.ylabel('mse')

plt.legend()

plt.show()