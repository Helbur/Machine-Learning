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
maxdegree = 10

# creating empty arrays for future errors and scores
MSE_test = np.zeros(maxdegree)
R2_test = np.zeros(maxdegree)

MSES_test = np.zeros(maxdegree)
MSES_scaled_test = np.zeros(maxdegree)

MSE_train = np.zeros(maxdegree)
R2_train = np.zeros(maxdegree)

MSE_scaled_test = np.zeros(maxdegree)
R2_scaled_test = np.zeros(maxdegree)

MSE_scaled_train = np.zeros(maxdegree)
R2_scaled_train = np.zeros(maxdegree)

#filling polydegree array
polydegree = np.zeros(maxdegree)

for i in range(1, maxdegree+1, 1):

	#creating X matrix on i polydegree
	X = create_X(x, y, i)

	#splitiing data
	X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

	#calculating beta by matrix inversion
	beta = np.linalg.pinv(X_train.T @ (X_train)) @ (X_train.T) @ (z_train)

	#making skl linreg fit to compare with
	clf = LinearRegression().fit(X_train, z_train)
	z_spred_test = clf.predict(X_test)

	#test and train prediction 
	z_pred_train = X_train @ beta
	z_pred_test = X_test @ beta

	#filling error and score arrays
	MSE_test[i-1] = Mean_Square_Error(z_pred_test, z_test)
	R2_test[i-1] = R2_Score_Function(z_pred_test, z_test)

	MSES_test[i-1] = Mean_Square_Error(z_spred_test, z_test)

	MSE_train[i-1] = Mean_Square_Error(z_pred_train, z_train)
	R2_train[i-1] = R2_Score_Function(z_pred_train, z_train)

	#filling polydegree arry
	polydegree[i-1] = i

	#printing and comparing error ands scroe values

	#print(MSE_test[i-1], R2_test[i-1])
	#print(MSE_train[i-1], R2_train[i-1])
	print(MSE_test[i-1], MSES_test[i-1])


#the same procedure with scaled data
for i in range(1, maxdegree+1, 1):
	X = create_X(x, y, i)

	X_scaled_train, X_scaled_test, z_scaled_train, z_scaled_test = train_test_split(X, z, test_size = 0.2 )

	scaler = StandardScaler(with_std=False)
	X_scaled = scaler.fit_transform(X_scaled_train)
	z_scaled = scaler.fit_transform(z_scaled_train)


	clf1 = LinearRegression().fit(X_scaled_train, z_scaled_train)
	z_scaled_spred_test = clf1.predict(X_scaled_test)

	beta_scaled = np.linalg.pinv(X_scaled_train.T @ (X_scaled_train)) @ (X_scaled_train.T) @ (z_scaled_train)

	z_scaled_pred_train = X_scaled_train @ beta_scaled
	z_scaled_pred_test = X_scaled_test @ beta_scaled

	MSE_scaled_test[i-1] = Mean_Square_Error(z_scaled_pred_test, z_scaled_test)
	R2_scaled_test[i-1] = R2_Score_Function(z_scaled_pred_test, z_scaled_test)

	MSES_scaled_test[i-1] = Mean_Square_Error(z_scaled_spred_test, z_scaled_test)

	MSE_scaled_train[i-1] = Mean_Square_Error(z_scaled_pred_train, z_scaled_train)
	R2_scaled_train[i-1] = R2_Score_Function(z_scaled_pred_train, z_scaled_train)

	print(MSE_scaled_test[i-1], MSES_scaled_test[i-1])


#poltting stuff
plt.figure(1)
plt.plot(polydegree, MSE_test, label='Test MSE')
plt.plot(polydegree, MSE_train, label='Train MSE')
plt.xlabel('Polynomial Fit Degree')
plt.ylabel('Mean Square Error')
plt.title('Not scaled data')
plt.legend()

plt.figure(2)
plt.plot(polydegree, R2_test, label='Test R2')
plt.plot(polydegree, R2_train, label='Train R2')
plt.xlabel('Polynomial Fit Degree')
plt.ylabel('R2 Error Function')
plt.title('Not scaled data')
plt.legend()

plt.figure(3)
plt.plot(polydegree, MSE_test, label='not scaled MSE')
plt.plot(polydegree, MSE_scaled_test, label='scaled MSE')
plt.xlabel('Polynomial Fit Degree')
plt.ylabel('Mean Square Error')
plt.title('not scaled vs scaled data')
plt.legend()
plt.show()