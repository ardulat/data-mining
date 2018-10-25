import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation

tt = pd.read_csv('data_lda_circular.txt', header=None)
X = tt.values[:, 0:2]
y = tt.values[:, 2]

# Loop for every depth
for depth in range(1, 11):
    clf = DecisionTreeClassifier(max_depth=depth)

    # Train data
    clf.fit(X, y)

    # Test data
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)
    y_predicted = clf.predict(X_test)
    err = X_test[y_predicted != y_test]
    if (len(err) == 0):
        print ('No errors found!')

    # Plot
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.scatter(err[:, 0], err[:, 1], color='red', edgecolors='k')
    plt.title('max_depth = %d' % depth)
    plt.show()
