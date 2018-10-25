import numpy as np

from numpy.linalg import inv


X = np.array([[1,1],
             [0,1]])
y = np.array([[1.5,1.9,2.7,3.1],
             [2.5,3.5,1.5, 3.3]])

parentheses = inv(np.dot(X, X.T))

right_side = np.dot(X.T, y)

result = np.dot(parentheses, right_side)

print result