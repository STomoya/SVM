'''
A Support Vector Machine implementation by Python.

This implementation solves the quadratic programming (QP) problem for optimization.
This implementation of SVM is not popular, because its cost for calculation.
Sequential Miminal Optimization (SMO) was created for this reason
and it reduces some calculation. (see svm_smo.py for the implemetation.)

Still, many SVMs that are implemented from scratch solves QP to optimize the model.
(Meaning there are bunch of articles in the Internet.)
And also the most simplest way for optimizing SVMs if you look at it from the mathematics.

NOTE: This implementation relise on "cvxopt" module,
        which is a convex optimization module,
        to solve the QP problem in optimization.
        (Couldn't find one that doesn't use this library...)

Most of the implementation follows the methematics presented in PRML[1].
It might be good to read the book (specifically Chapter 7.) for basic understanding of the SVM.
I believe it's easier than the other chapters...

[REFERENCE]
[1] Bishop, C.,
    "Pattern Recognition and Machine Learning",
    Springer,
    2006

[CODE INSPIRED BY]
 1) "Support Vector Machine: Python implementation using CVXOPT"
    https://xavierbourretsicotte.github.io/SVM_implementation.html
 2) "線形SVMの理論と実装"
    https://satopirka.com/2018/12/theory-and-implementation-of-linear-support-vector-machine/
'''

import numpy as np
import cvxopt

class QPSVC():
    def __init__(self, kernel, C=1., gamma=10):
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

    def fit(self, X, y):
        N, n_features = X.shape
        _Q = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                _Q[i, j] = y[i] * y[j] * self.K(X[i], X[j])
        Q = cvxopt.matrix(_Q)
        p = cvxopt.matrix(-np.ones(N))
        G = cvxopt.matrix(np.vstack((-np.eye(N), np.eye(N))))
        h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N) * self.C)))
        A = cvxopt.matrix(y, (1, N), 'd')
        b = cvxopt.matrix(0.0)

        print(A)
        solution = cvxopt.solvers.qp(Q, p, G, h, A, b)

        alpha = np.ravel(solution['x'])

        sv = alpha > 1e-5
        ind = np.arange(len(alpha))[sv]
        self.a = alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        
        self.w = np.dot(self.sv.T, self.a * self.sv_y)
        # self.b = np.mean(self.sv_y - np.dot(self.sv, self.w))
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * _Q[ind[n],sv])
        self.b /= len(self.a)

    def decision_function(self, X):
        if self.K == self.__linear_kernel:
            return np.dot(X, self.w) + self.b
        else:
            scores = []
            for x in X:
                scores.append(sum([a * sv_y * self.K(x, sv) for a, sv_y, sv in zip(self.a, self.sv_y, self.sv)]))
            scores = np.array(scores)
            return scores

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
    SVC = QPSVC(kernel='rbf', gamma=1)
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
    plt.savefig('output_qp.png')