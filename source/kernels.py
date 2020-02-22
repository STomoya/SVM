
import numpy as np

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