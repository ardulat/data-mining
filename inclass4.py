import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm

tt = pd.read_csv('simple_bin_classifier.txt', header=None)
X = tt.values[:, 0:2]
y = tt.values[:, 2]

# clf = svm.SVC(kernel='rbf', gamma=1.0)
clf = svm.SVC(kernel='linear')

clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.plot(X[clf.support_,0], X[clf.support_,1], '.', color='b')
plt.show()