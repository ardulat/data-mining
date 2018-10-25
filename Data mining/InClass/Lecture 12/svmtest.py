#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import minimize
from sklearn import metrics, neighbors
from sklearn import cross_validation

from sklearn import svm, datasets

# Loading data for previous examples
tt = pd.read_csv("data_lda_circular.txt",header=None)
X = tt.values[:,0:2]
y = tt.values[:,2]


C = 1.0

clf = svm.SVC(kernel='rbf', gamma=1, C=C)
#clf = svm.SVC(kernel='linear', C=C)

clf.fit(X, y)

plot_step = 0.01

x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.PuBu)

err = clf.predict(X) != y

ci="black"
for i in range(0,4):
    if i == 2:
        ci="gray"
    plt.plot(X[y==i,0],X[y==i,1],'o',c=ci)
    plt.axis("image")

plt.plot(X[err, 0], X[err,1], 'ro')
plt.title("svmsvm_figure")

plt.show()
