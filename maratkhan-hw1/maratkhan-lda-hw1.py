import numpy as np
import pandas as pd
import csv

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix


tt = pd.read_csv("dm-hw-m-train.txt", header=None)

index = tt.values[:, 0]
X = tt.values[:,1:4]
y = tt.values[:, 4]

# Performing cross-validation
clf = LinearDiscriminantAnalysis()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
score = (tp+tn)/(tn+fp+fn+tp) # accuracy score

print ('LDA accuracy score: %f' % score)

# Applying on real test data
clf = LinearDiscriminantAnalysis()

clf.fit(X, y)

test = pd.read_csv("dm-hw-m-test-dist.txt", header=None)

index_test = test.values[:, 0]
X_test = test.values[:, 1:4]

y_pred = clf.predict(X_test)

with open('maratkhan-lda-hw1.csv', 'w') as f:
    writer = csv.writer(f)

    for i in range(len(index_test)):
        writer.writerow([index_test[i], y_pred[i]])

    f.close()