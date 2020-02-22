
import numpy as np
import time

from binary_svm import BinarySVC

class SVC:
    '''
    An implementation of SVM for multiclass classification based on One versus Rest.

    NOTE: The problem of adapting SVMs to multiclass tasks is incomplete.
            This implementation only follows the most used method,
            which is "return the label that showed the highest value by the decision function".
            The separate SVMs will learn different classification problems,
            and there might not be any significance in comparing the outputs.
            Also the percentage of the labels will be unbalanced when separating the targets to one and others.
    '''
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

        chooses predicted label by the maximum value of the decision function.
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
    from kernels import LinearKernel, GaussianKernel
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
    