
import numpy as np
import time

from binary_svm import BinarySVC

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
    def __init__(self, kernel, C):
        self.C = C
        self.kernel = kernel
    
    def fit(self, X, t, tol_conv=1e-6, tol_sv=1e-9, show_time = False):
        '''
        fit SVM to input data

        if includes only two classes -> binary-class classification
        if includes multiple classes -> classification based on One versus Rest
        if only one class exists     -> raise exception
        '''
        self.class_ = list(set(t))
        if len(self.class_) == 2:
            self.svc = BinarySVC(kernel=self.kernel, C=self.C)
            self.svc.fit(X, t, tol_conv, tol_sv, show_time)
        elif len(self.class_) > 2:
            self.multiclass_fit(X, t, tol_conv, tol_sv, show_time)
        else:
            raise Exception('There must be more than two classes in the dataset.')
    
    def multiclass_fit(self, X, t, tol_conv=1e-6, tol_sv=1e-9, show_time=False):
        '''
        SVM multi-class classification based on One versus Rest.
        '''
        # generate targets for One versus Rest
        self.ovr_ts = {}
        # generate binary SVCs for each target
        self.svcs = {}
        for c in self.class_:
            self.ovr_ts[c] = np.array([1 if one == c else -1 for one in t])
            self.svcs[c] = BinarySVC(kernel=self.kernel, C=self.C)

        # fit to each SVCs
        for c in self.class_:
            self.svcs[c].fit(X, self.ovr_ts[c], tol_conv, tol_sv, show_time)
    
    def predict(self, X):
        '''
        return an array of predicted labels

        if includes multiple classes -> multiclass_prediction()
        '''
        if len(self.class_) == 2:
            return self.svc.predict(X)
        else:
            return self.multiclass_predict(X)

    def multiclass_predict(self, X):
        '''
        prediction function for mutli-class classification
        '''
        # get value of decision function for each target
        val_dec_funcs = []
        for c in self.class_:
            val_dec_funcs.append(self.svcs[c].decision_function(X))
        
        # find index of the highest value
        indexs = np.argmax(val_dec_funcs, axis=0)
        
        # convert index to class
        predicted_ts = [self.class_[index] for index in indexs]

        return predicted_ts

if __name__=='__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report

    iris = datasets.load_iris()
    X = iris['data']
    y = iris['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    svc = SVC(C=0.1, kernel=GaussianKernel(theta=0.5))
    svc.fit(X_train, y_train)

    prediction = svc.predict(X_test)
    print('confusion matrix')
    print(confusion_matrix(y_test, prediction))
    print('classification report')
    print(classification_report(y_test, prediction))
    