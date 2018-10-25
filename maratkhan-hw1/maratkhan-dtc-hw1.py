import numpy as np
import pandas as pd
import csv

from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix


tt = pd.read_csv("dm-hw-m-train.txt", header=None)

index = tt.values[:, 0]
X = tt.values[:,1:4]
y = tt.values[:, 4]

depth = range(1, 1000)
score_list = []
depth_list = []

for i in depth:
    clf = DecisionTreeClassifier(max_depth=i)
    # cross validation
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)
    clf.fit (X_train, y_train)
    y_pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    score = (tp+tn)/(tn+fp+fn+tp) # accuracy score
    score_list.append(score)
    depth_list.append(i)

best_depth = depth_list[np.argmax(score_list)]
print ('Best score obtained (default splitter): %f' % max(score_list))
print ('Best depth chosen (default splitter): %f' % best_depth)

score_list_random = []
depth_list_random = []

for i in depth:
    clf = DecisionTreeClassifier(max_depth=i, splitter='random')
    # cross validation
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)
    clf.fit (X_train, y_train)
    y_pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    score = (tp+tn)/(tn+fp+fn+tp) # accuracy score
    score_list_random.append(score)
    depth_list_random.append(i)

best_depth_random = depth_list_random[np.argmax(score_list_random)]
print ('Best score obtained (random splitter): %f' % max(score_list_random))
print ('Best depth chosen (random splitter): %f' % best_depth_random)

# Choosing the best depth and splitter
depth_couple = [best_depth, best_depth_random]
score_couple = [max(score_list), max(score_list_random)]
splitter_couple = ['best', 'random']
i = np.argmax(score_couple)
depth_chosen = depth_couple[i]
splitter_chosen = splitter_couple[i]
print ('depth: %d, splitter: %s' % (depth_chosen, splitter_chosen))

# Testing
test = pd.read_csv("dm-hw-m-test-dist.txt", header=None)

index_test = test.values[:, 0]
X_test = test.values[:, 1:4]

clf = DecisionTreeClassifier(max_depth=depth_chosen, splitter=splitter_chosen)
clf.fit (X, y)

y_pred = clf.predict(X_test)

with open('maratkhan-dtc-hw1.csv', 'w') as f:
    writer = csv.writer(f)

    for i in range(len(index_test)):
        writer.writerow([index_test[i], y_pred[i]])

    f.close()