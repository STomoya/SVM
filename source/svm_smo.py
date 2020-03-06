'''
A Support Vector Machine implementation by Python.

This implementation uses the Sequential Miminal Optimization (SMO) algorithm
for solving the quadratic programming problem that appears in the optimization of the SVM.
In many implementation of the SVM uses SMO for optimization. (Like scikit-learn. (To be more percise, LIBSVM, becauce scikit-learn uses LIBSVM for optimization.))
Still for implementing from scratch, it might be difficult to understand the mathematics of the algorithm...

This implementation follows the algorithm in the work of Change et al. [1].
The working set is chosen by finding the maximal violating pair,
which choses one data point from each class, that most violate the KKT conditions.
(There are other ways to choses the working set.)

[REFERENCE]
[1] Bottou, L., Lin, C.,
    "Support Vector Machine Solvers",
    2007,
    https://www.csie.ntu.edu.tw/~cjlin/papers/bottou_lin.pdf

[CODE INSPIRED BY]
 1) "PRML第7章のサポートベクターマシン(SVM)をPythonで実装"
    https://qiita.com/amber_kshz/items/6a9f8b6dd857edffce58
'''

import numpy as np

class SMOSVC():
    def __init__(self, kernel, C, gamma=10):
        self.gamma = gamma
        self.C = float(C)
        self.gamma = gamma

        if kernel == 'linear':
            self.K = self.__linear_kernel
        elif kernel == 'rbf':
            self.K = self.__gaussian_kernel
        else:
            raise Exception('No such kernel as {}'.format(kernel))

    def __linear_kernel(self, xi, xj):
        return np.dot(xi, xj)
    
    def __gaussian_kernel(self, xi, xj):
        dif = xi - xj
        return np.exp(-self.gamma * np.dot(dif, dif))

    def fit(self, X, y, tol_conv=1e-6, tol_sv=1e-9):
        N, n_features = X.shape
        
        a = np.zeros(N)
        g = np.ones(N)

        while True:
            yxg = y * g
            
            iup  = np.argmax([yxg[index] if ((a[index] < self.C - tol_sv and y[index] == 1) or (a[index] > tol_sv and y[index] == -1)) else -float('inf') for index in range(N)])
            ilow = np.argmin([yxg[index] if ((a[index] < self.C - tol_sv and y[index] == -1) or (a[index] > tol_sv and y[index] == 1)) else float("inf") for index in range(N)])

            if yxg[iup] <= yxg[ilow] + tol_conv:
                self.b = (yxg[iup] + yxg[ilow]) / 2
                break

            else:
                A = self.C - a[iup] if y[iup] == 1 else a[iup]
                B = a[ilow] if y[ilow] == 1 else self.C - a[ilow]
                C = (y[iup]*g[iup] - y[ilow]*g[ilow]) / (self.K(X[iup], X[iup]) - 2*self.K(X[iup], X[ilow]) + self.K(X[ilow], X[ilow]))
                lambda_ = min(A, B, C)
                a[iup]  = a[iup]  + lambda_*y[iup]
                a[ilow] = a[ilow] - lambda_*y[ilow]
                for index in range(N):
                    g[index] = g[index] - lambda_*y[index]*(self.K(X[index], X[iup]) - self.K(X[index], X[ilow]))
        
        sv_indices = a > tol_sv
        self.a = a[sv_indices]
        self.sv = X[sv_indices]
        self.sv_y = y[sv_indices]
    
    def decision_function(self, X):
        scores = []
        for x in X:
            scores.append(sum([np.dot(self.K(x, sv), a*y) for sv, a, y in zip(self.sv, self.sv_y, self.a)]))
        scores = np.array(scores) + self.b
        return scores
    
    def predict(self, X):
        return np.sign(self.decision_function(X))

if __name__=='__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    # creating random dataset (2 features for easy plotting)
    # X, Y = datasets.make_classification(
    #                         n_samples=500,
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
    y_ = np.array([-1 if y == 0 else 1 for y in Y])

    # data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y_, random_state=256, test_size=0.2)

    # training
    SVC = SMOSVC(kernel='rbf', C=1, gamma=1)
    SVC.fit(X_train, y_train)

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
    plt.savefig('output_smo.png')