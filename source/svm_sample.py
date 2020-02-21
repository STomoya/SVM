
# code from https://qiita.com/amber_kshz/items/6a9f8b6dd857edffce58

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import time

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# np.random.seed(42)

class LinearKernel:
    def __init__(self, theta=None):
        pass
    
    def __call__(self, X, Y):
        '''
        This method calculates kernel matrix, or a single scalar, or a vector, depending on the input.
        The kernel function is assumed to be the linear kernel function
        
        Parameters
        ----------
        X, Y : 2-D numpy array, or 1-D numpy array
            numpy array representing input points. 
            If X (resp. Y) is a 2-D numpy array, X[n, i] (resp. Y[n, i]) represents the i-th element of n-th point in X (resp Y).
            If X (resp. Y) is a 1-D numpy array, it is understood that it represents one point, where X[i] (resp Y[i]) reprensets the i-th element of the point.
        
        Returns
        ----------
        K : numpy array
            If both X and Y are both 2-D numpy array, K is a 2-D numpy array, with shape = (len(X), len(Y)), and K[i, j] stands for k(X[i], Y[j]).
            If X is a 2-D numpy array and Y is a 1-D numpy array, K is a 1-D numpy array, where 
        '''
        return X @ Y.T

class GaussianKernel:
    def __init__(self, theta):
        self.theta = theta
        
    def __call__(self, X, Y):
        '''
        This method calculates kernel matrix, or a single scalar, or a vector, depending on the input.
        The kernel function is assumed to be the Gaussian kernel function
        
        Parameters
        ----------
        X, Y : 2-D numpy array, or 1-D numpy array
            numpy array representing input points. 
            If X (resp. Y) is a 2-D numpy array, X[n, i] (resp. Y[n, i]) represents the i-th element of n-th point in X (resp Y).
            If X (resp. Y) is a 1-D numpy array, it is understood that it represents one point, where X[i] (resp Y[i]) reprensets the i-th element of the point.
        
        Returns
        ----------
        K : numpy array
            If both X and Y are both 2-D numpy array, K is a 2-D numpy array, with shape = (len(X), len(Y)), and K[i, j] stands for k(X[i], Y[j]).
            If X is a 2-D numpy array and Y is a 1-D numpy array, K is a 1-D numpy array, where 
        '''
        if (X.ndim == 1) and (Y.ndim == 1):
            tmp = np.linalg.norm(X - Y)**2
        elif ((X.ndim == 1) and (Y.ndim != 1)) or ((X.ndim != 1) and (Y.ndim == 1)):
            tmp = np.linalg.norm(X - Y, axis=1)**2
        else:
            tmp = np.reshape(np.sum(X**2,axis=1), (len(X), 1)) + np.sum(Y**2, axis=1)  -2 * (X @ Y.T)
        K = np.exp(- tmp/(2*self.theta**2))
        return K

class SVC:
    def __init__(self, kernel, C=1.0):
        self.C = C
        self.kernel = kernel

    def fit(self, X, t, tol_conv=1e-6, tol_sv=1e-9, show_time = False):
        '''
        This method fits the classifier to the training data
        
        Parameters
        ----------
        X : 2-D numpy array
            Array representing training input data, with X[n, i] being the i-th element of n-th point in X.
        t : 1-D numpy array
            Array representing training label data. Each component should be either 1 or -1
        tol_conv : float
            Tolerance for numerical error when judging the convergence.
        tol_sv : float
            Tolerance for numerical error when judging the support vectors.
        show_time : boolean
            If True, training time and the number of iteration is shown.
        
        '''
        N = len(t)
        
        a = np.zeros(N)
        g = np.ones(N)
        
        nit = 0
        time_start = time.time()
        while True:
            vals = t*g
            i = np.argmax([ vals[n] if ( (a[n] < self.C - tol_sv and t[n]==1) or (a[n] > tol_sv and t[n]==-1)) else -float("inf") for n in range(N) ])
            j = np.argmin([ vals[n] if ( (a[n] < self.C - tol_sv and t[n]==-1) or (a[n] > tol_sv and t[n]==1)) else float("inf") for n in range(N) ] )
            if vals[i] <= vals[j] + tol_conv:
                self.b = (vals[i] + vals[j])/2
                break
            else:
                A = self.C - a[i] if t[i] == 1 else a[i]
                B = a[j] if t[j] == 1 else self.C - a[j]
                lam = min(A, B, (t[i]*g[i] - t[j]*g[j])/(self.kernel(X[i], X[i]) - 2*self.kernel(X[i], X[j]) + self.kernel(X[j], X[j])) )
                a[i] = a[i] + lam*t[i]
                a[j] = a[j] - lam*t[j]
                g = g - lam*t*( self.kernel(X, X[i]) - self.kernel(X, X[j]) )
            nit += 1
        time_end = time.time()
        if show_time:
            print(f"The number of iteration : {nit}")
            print(f"Learning time : {time_end-time_start} seconds")
        ind_sv = np.where( a > tol_sv)
        self.a = a[ind_sv]
        self.X_sv = X[ind_sv]
        self.t_sv = t[ind_sv]

    def decision_function(self, X):
        '''
        This method returns the value of the decision function for the given input X.
        
        Parameters
        ----------
        X : 2-D numpy array
            Array representing input data, with X[n, i] being the i-th element of n-th point in X.
        
        Returns
        ----------
        val_dec_func : 1-D numpy array
            Array representing the value of the decision function
        
        '''
        val_dec_func = self.kernel(X, self.X_sv) @ (self.a*self.t_sv) + self.b
        return val_dec_func 
    
    def predict(self,X):
        '''
        This method returns the predicted label for the given input X.
        
        Parameters
        ----------
        X : 2-D numpy array
            Array representing input data, with X[n, i] being the i-th element of n-th point in X.
        
        Returns
        ----------
        pred : 1-D numpy array
            Array representing the predicted label
        
        '''
        pred = np.sign(self.decision_function(X))
        return pred

def get_meshgrid(x, y, nx, ny, margin=0.1):
    x_min, x_max = (1 + margin) * x.min() - margin * x.max(), (1 + margin) * x.max() - margin * x.min()
    y_min, y_max = (1 + margin) * y.min() - margin * y.max(), (1 + margin) * y.max() - margin * y.min()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    return xx, yy

def plot_result(ax, clf, xx, yy, X, t, x, plot_sv=True, plot_decision_function=False):
    if plot_decision_function:
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.7)
    ax.scatter(X[:,0], X[:,1], c=t, edgecolor='k')
    ax.scatter(x[:,0], x[:,1])
    if plot_sv:
        ax.scatter(clf.X_sv[:,0], clf.X_sv[:,1], c=clf.t_sv, marker='s', s=100, edgecolor='k')
    plt.savefig('svm.png')

from sklearn import datasets

X, t = datasets.make_moons(n_samples = 200, noise = 0.1)
t = 2*t-1
plt.scatter(X[:,0], X[:,1], c=t, edgecolor='k')
plt.savefig('data.png')

xx, yy = get_meshgrid(X[:, 0], X[:, 1], nx=300, ny=300, margin=0.1)
x = np.random.rand(1, 2)

kernel = LinearKernel()
kernel = GaussianKernel(theta=0.5)

linsvc = SVC(C=10.0, kernel=kernel)
linsvc.fit(X, t, show_time=True)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1,2,1)
ax.set_title("decision function")
plot_result(ax=ax, clf=linsvc, xx=xx, yy=yy, X=X, t=t, x=x, plot_decision_function=True)
ax = fig.add_subplot(1,2,2)
ax.set_title("prediction")
plot_result(ax=ax, clf=linsvc, xx=xx, yy=yy, X=X, t=t, x=x, plot_decision_function=False)