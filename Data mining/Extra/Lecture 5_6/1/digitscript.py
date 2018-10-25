#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

from sklearn import datasets

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# returns left/right symmetry of an 8x8 image
def symlr(t):
    s = np.fliplr(t)
    y = t - s
    val = np.sum(y[:, 0:3])
    return val

# returns up/down symmetry of an 8x8 image
def symud(t):
    s = np.flipud(t)
    y = t - s
    val = np.sum(y[0:3, :])
    return val

# load the digits data
digits = datasets.load_digits()

n_samples = len(digits.images)
print(digits.images.shape)

# calculate symmetry features for all images in the data set
lr = []
ud = []

for im in digits.images:
    lr.append(symlr(im))
    ud.append(symud(im))

LR = np.array(lr)
UD = np.array(ud)

B = np.hstack((LR[:, np.newaxis], UD[:, np.newaxis]))
print(B.shape)

# compare two particular digits
ii = 0
jj = 7

I = digits.target == ii
J = digits.target == jj

X = np.vstack((B[I, :], B[J, :]))
y = np.hstack((digits.target[I], digits.target[J]))

print(X.shape)
print(y.shape)

clf = LinearDiscriminantAnalysis()

clf.fit(X, y)

err = clf.predict(X) != y

for i in [ii, jj]:
    II = digits.target == i
    plt.plot(B[II, 0], B[II, 1], 'o', label=str(i))

plt.plot(X[err, 0], X[err, 1], 'ro')

f = 0
h = scipy.stats.kruskal(B[digits.target == ii, f], B[digits.target == jj, f])
print(h)

plt.figure()
plt.boxplot([B[digits.target == ii, f], B[digits.target == jj, f]])

#plt.legend()
#plt.plot(LR[J], UD[J], 'go')

plt.figure()
plt.plot(y)
plt.plot(clf.predict(X))

plt.show()

# data = digits.images.reshape((n_samples, -1))
# #print(data.shape)
#
# X = data[digits.target == 2, :]
# Y = data[digits.target == 8, :]
#
# feature1 = data[:, 35]
# feature2 = data[:, 37]
#
# R = np.corrcoef(data, rowvar=0)
#
# e = np.random.rand(n_samples)
# #print((feature1 + e).shape)
#
# r = scipy.stats.pearsonr(feature1, feature2)
# #print(r)
#
# # plt.figure(1)
# # plt.plot(feature1 + e, feature2 + np.random.rand(n_samples), 'bo')
# #
# # #plt.imshow((Y[30, :]).reshape((8, 8)), interpolation="nearest")
# # #plt.show()
# #
# # #plt.imshow(R, interpolation="nearest")
# # #plt.show()
# #
# # plt.figure(2)
# # ex = np.random.rand(len(X[:, 35]))
# # ey = np.random.rand(len(Y[:, 35]))
# # plt.plot(X[:, 35] + ex, X[:, 36] + np.random.rand(len(X[:, 35])), 'bo')
# # plt.plot(Y[:, 35] + ey, Y[:, 36] + np.random.rand(len(Y[:, 35])), 'ro')
# #
# #
# # plt.figure(3)
# # plt.imshow(R, interpolation="nearest")
# #
# # #plt.plot(data[:,31])
# # plt.show()