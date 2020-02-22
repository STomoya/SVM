'''
[CODE] https://qiita.com/amber_kshz/items/6a9f8b6dd857edffce58
[REFERENCE] https://www.csie.ntu.edu.tw/~cjlin/papers/bottou_lin.pdf
'''

import numpy as np
import time

class BinarySVC:
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
