import numpy as np
import pandas as pd
import csv

from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix


tt = pd.read_csv("dm-hw-m-train.txt", header=None)

index = tt.values[:, 0]
X = tt.values[:,1:4]
y = tt.values[:, 4]

gamma = np.linspace(0.01, 100, num=1000)
score_list = []
gamma_list = []

for i in gamma:
    clf = SVC(kernel='rbf', gamma=i)    
    # cross validation
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)
    clf.fit (X_train, y_train)
    y_pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    score = (tp+tn)/(tn+fp+fn+tp) # accuracy score
    score_list.append(score)
    gamma_list.append(i)

best_gamma = gamma_list[np.argmax(score_list)]
print ('Best score obtained: %f' % max(score_list))
print ('Best gamma chosen: %f' % best_gamma)

# Testing
test = pd.read_csv("dm-hw-m-test-dist.txt", header=None)

index_test = test.values[:, 0]
X_test = test.values[:, 1:4]

clf = SVC(kernel='rbf', gamma=best_gamma)
clf.fit (X, y)

y_pred = clf.predict(X_test)

with open('maratkhan-svm-hw1.csv', 'w') as f:
    writer = csv.writer(f)

    for i in range(len(index_test)):
        writer.writerow([index_test[i], y_pred[i]])

    f.close()