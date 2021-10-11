import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import random, seed
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(0)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# Datapoints
x = np.random.uniform(0.0, 1.0, 200); y = x
z = FrankeFunction(x,y)

# Gaussian noise centered at 0 with sigma=0.1
zNoise = FrankeFunction(x,y) + np.random.normal(0, 0.1, size=z.shape)

# Feature matrix construction and data splitting for different polynomial fits
x = (x - np.mean(x))
y = (y - np.mean(y))
# Stores MSEs in a list for up to 9 degree polynomials(Ex. 2)
MSETest = []
MSETrain = []
# for i in range(1, 9+1):
#     X = np.vstack((x,y)).T
#     poly = PolynomialFeatures(degree=i)
#     X = poly.fit_transform(X)
#     X_train, X_test, z_train, z_test = train_test_split(X, zNoise, test_size=0.2)
#     linreg = LinearRegression()
#     linreg.fit(X_train, z_train)
#     zpredTest = linreg.predict(X_test)
#     zpredTrain = linreg.predict(X_train)
#     # Mean Squared Error
#     MSETest.append(mean_squared_error(z_test, zpredTest))
#     MSETrain.append(mean_squared_error(z_train, zpredTrain))
#     #print(f"MSE for deg={i} polynomial: ", mean_squared_error(z_test, zpred))
#     # R2 score
#     #print(f"R2 score for deg={i} polynomial: ", r2_score(z_test, zpred))

#---------MSE PLOT-------#
plt.plot(MSETest)
plt.plot(MSETrain)
plt.xlabel("Complexity")
plt.ylabel("Mean Squared Error")
plt.legend(["Test sample", "Training sample"])

#---------FRANKE PLOT---------#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# x = np.arange(0.0, 1.0, 0.05); y = x
# x, y = np.meshgrid(x,y)

# z = FrankeFunction(x, y)
# zNoise = z + np.random.normal(0, 0.1, size=z.shape)
# #surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,\
# #                        linewidth=0, antialiased=False)
# ax.scatter(x,y,zNoise)
# ax.set_zlim(-0.10, 1.40)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# plt.title("Noisy Franke Function")
# plt.xlabel("x")
# plt.ylabel("y")
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()