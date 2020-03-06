'''
A Support Vector Machine implementation by Python.

This implementation uses the online training method to optimize the SVM.
Stochastic Gradient Descent (SGD) is used to optimize the weights.
Online training is recently used for training SVMs on large datasets.
Still the accuracy is a bit lower than using the SMO or solving the quadratic programming (QP) problem in optimization.

This implementation uses the algorithm of Pegasos[1],
a method for training the SVM using SGD.

[REFERNCE]
[1] Shalev-Shwartz, S., Singer, Y., Srebro, N. et al.
    "Pegasos: primal estimated sub-gradient solver for SVM."
    Math. Program. 127, 3–30 (2011).
    https://doi.org/10.1007/s10107-010-0420-4

[CODE INSPIRED BY]
 1) "確率的勾配降下法によるSVMの実装"
    https://qiita.com/sz_dr/items/763239df283c7e96be99
 2) "Implementing PEGASOS: Primal Estimated sub-GrAdient SOlver for SVM, Logistic Regression and Application in Sentiment"
    https://sandipanweb.wordpress.com/2018/04/29/implementing-pegasos-primal-estimated-sub-gradient-solver-for-svm-using-it-for-sentiment-classification-and-switching-to-logistic-regression-objective-by-changing-the-loss-function-in-python/
'''

import numpy as np

class PegasosSVC():
    '''
    Pegasos SVC.
    
    Does not support kernel.
    '''
    def __init__(self, lambda_):
        self.lambda_ = lambda_
        self.w = None
    
    def fit(self, X, y, epochs):
        N, n_features = X.shape
        self.w = np.zeros(n_features)
        for t in range(1, epochs+1):
            eta = 1. / (self.lambda_*t)
            it = np.random.choice(N, 1)[0]
            self.__update_w(X[it], y[it], eta)

    def __update_w(self, xi, yi, eta):
        d = 1 - yi * np.dot(xi, self.w)
        if max(0, d) == 0:
            self.w = (1 - eta*self.lambda_) * self.w
        else:
            self.w = (1 - eta*self.lambda_) * self.w + eta * yi * xi
    
    def decision_function(self, X):
        cost = np.dot(X, self.w)
        return cost
    
    def predict(self, X):
        cost = self.decision_function(X)
        predicted = np.sign(cost)
        return predicted

class KernelizedPegasosSVC():
    '''
    kernelized Pegasos SVC

    Supports kernels.
    '''
    def __init__(self, lambda_, kernel, gamma=10):
        self.lambda_ = lambda_
        self.gamma = gamma

        if kernel == 'linear':
            self.K = self.__linear_kernel
        elif kernel == 'rbf':
            self.K = self.__gaussian_kernel
        else:
            raise Exception('No such kernel as {}'.format(kernel))

        self.alpha = None
    
    def __linear_kernel(self, xi, xj):
        return np.dot(xi, xj)
    
    def __gaussian_kernel(self, xi, xj):
        dif = xi - xj
        return np.exp(-self.gamma * np.dot(dif, dif))

    def fit(self, X, y, epochs):
        self.epochs = epochs
        self.X = X
        self.y = y
        N, n_features = self.X.shape
        self.alpha = np.zeros(N)
        for t in range(1, epochs+1):
            eta = 1. / (self.lambda_*t)
            it = np.random.choice(N, 1)[0]
            self.__update_alpha(it, eta)

    def __update_alpha(self, it, eta):
        xi = self.X[it]
        yi = self.y[it]
        d = yi * eta * sum([alpha_j * yi * self.K(xi, xj) for xj, alpha_j in zip(self.X, self.alpha)])
        if d < 1.:
            self.alpha[it] += 1
    
    def decision_function(self, X):
        eta = 1. / (self.lambda_ * self.epochs)
        cost = []
        for x in X:
            cost.append(eta * sum([alpha_j * yj * self.K(xj, x) for xj, alpha_j, yj in zip(self.X, self.alpha, self.y)]))
        return cost
    
    def predict(self, X):
        predicted = np.sign(self.decision_function(X))
        return predicted

if __name__=='__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    # creating random dataset (2 features for easy plotting)
    # X, Y = datasets.make_classification(random_state=16,
    #                         n_samples=50,
    #                         n_features=2, 
    #                         n_redundant=0, 
    #                         n_informative=1,
    #                         n_clusters_per_class=1,
    #                         n_classes=2)
    
    X, Y = datasets.make_moons(n_samples=500, noise=0.1)

    # visualize dataset
    plt.figure(figsize=(8, 7))
    plt.title("make_classification : n_features=2  n_classes=2")
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, s=25, edgecolor='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig('data.png')

    # labels to -1 or 1
    y_ = [-1 if y == 0 else 1 for y in Y]

    # data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y_, random_state=256, test_size=0.2)

    # training
    SVC = PegasosSVC(lambda_=0.1)
    SVC = KernelizedPegasosSVC(lambda_=0.1, kernel='rbf', gamma=10)
    SVC.fit(X_train, y_train, epochs=1000)

    # evaluate
    prediction = SVC.predict(X_test)
    print(classification_report(y_test, prediction))
    
    # plot boundary
    # get meshgrid
    margin = 0.1
    x1 = X[:, 0]
    x2 = X[:, 1]
    nx = 100
    ny = 100
    x_min, x_max = (1 + margin) * x1.min() - margin * x1.max(), (1 + margin) * x1.max() - margin * x1.min()
    y_min, y_max = (1 + margin) * x2.min() - margin * x2.max(), (1 + margin) * x2.max() - margin * x2.min()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    Z = SVC.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # plot
    plt.contourf(xx, yy, Z, alpha=0.7)
    plt.savefig('output_sgd.png')