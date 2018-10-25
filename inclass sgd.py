import numpy as np
import pandas as pd


tt = pd.read_csv('SGD Dataset.txt', header=None)

x = tt.values[:, 0:6]
y = tt.values[:, 6:12]

A = np.random.rand(6, 6)
iterations = 0
alpha = 0.00001
y = y.transpose()

while True:
    y1 = np.dot(A, x.transpose())

    delta = np.dot((y - y1), x)
    iterations += 1

    # print "%.6f" % np.linalg.det(delta)

    if abs(np.linalg.det(delta)) < 0.000000000001:
        break
    # e = (y - y1)
    # e = np.dot(e, e.transpose())

    A = A + alpha * delta


print iterations
print A
y1 = np.dot(A, x.transpose())
print y
print ""
print y1
e = y - y1
print ""
print e





