import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd
from sklearn.model_selection import train_test_split
np.random.seed(1)

def FrankeFunction(x,y):
    return 0.75*np.exp(-(9*x-2)**2/4. -(9*y-2)**2/4.) + \
           0.75*np.exp(-(9*x+1)**2/49.-(9*y+1)**2/10.) + \
           0.50*np.exp(-(9*x-7)**2/4. -(9*y-3)**2/4.) - \
           0.20*np.exp(-(9*x-4)**2 - (9*y-7)**2)

N = 100
x = np.random.uniform(0.0, 1.0, N)
y = np.random.uniform(0.0, 1.0, N)
X,Y = np.meshgrid(x,y)

def MSE(z, zpred):
    temp = (z-zpred) @ (z-zpred)
    return temp/float(N-1)

def R_sq(z, zpred):
    zmean = np.mean(z)
    upper = (z-zpred) @ (z-zpred)
    lower = (z-zmean) @ (z-zmean)
    return 1-upper/lower

def OLS(X, z):
    # using the Moore-Penrose pseudoinverse
    # accepts design matrix (X) and z-data
    # returns betas
    return np.linalg.pinv(X.T @ X) @ X.T @ z

# Ordinary Least Squares regression
# deg 2 poly fit
def construct_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2) # Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)
    return X

def construct_grid_plot(x, y, betas, n):
    # Works similar to construct_X but with meshgrids
    # This provides a nice visualization of the fit
    # sews together betas with meshes
    lst = [] #stores monomials
    lst.append(np.ones((20,20))*betas[0])
    j = 1
    for i in range(1,n+1):
        #q = int((i)*(i+1)/2)
        for k in range(i+1):
            lst.append(betas[j]*(x**(i-k))*(y**k))
            j += 1
    return np.sum(np.array(lst), axis=0)

polydeg = 4
X = construct_X(x, y, polydeg)
z = FrankeFunction(X[:,1], X[:,2]) + np.random.normal(0,0.03,N)

X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2, random_state=42)
betas = OLS(X_train, z_train)
zpred_train = X_train @ betas
zpred_test = X_test @ betas
print("MSE train: " + str(MSE(z_train, zpred_train)))
print("MSE test: " + str(MSE(z_test, zpred_test)))
print("-----------------------------------")
print("R2 train: " + str(R_sq(z_train, zpred_train)))
print("R2 test: " + str(R_sq(z_test, zpred_test)))

fig = plt.figure()
ax = fig.gca(projection='3d')
x = np.arange(0.0, 1.0, 0.05); y = x
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)
zNoise = z + np.random.normal(0, 0.03, size=z.shape)
#surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,\
#                        linewidth=0, antialiased=False)
ax.scatter(x,y,zNoise)
zPredGrid = construct_grid_plot(x, y, betas, polydeg)
ax.plot_surface(x, y, zPredGrid, color='red', alpha=0.4) #plot fit
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.view_init(elev=10., azim=25.)
plt.title("Noisy Franke Function, polynomial degree = " + str(polydeg))
plt.xlabel("x")
plt.ylabel("y")
#fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig(f"OLSfit_polynomial_degree_{polydeg}.jpeg")
plt.show()