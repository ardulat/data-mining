import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, SGDRegressor


tt = pd.read_csv('dm-final-train.txt', header=None)
test = pd.read_csv('dm-final-testdist.txt', header=None)

index = tt.values[:, 0]
X = tt.values[:, 1:101]
y = tt.values[:, 101:121]
indexTest = test.values[:, 0]
XTest = test.values[:, 1:]
# print(XTest.shape)
# XTest = XTest[np.newaxis, :]
# print(XTest.shape)

# svr = SVR(kernel='rbf', gamma=0.1)
# svr.fit(X, y)
# print(svr)

clf = LinearRegression()
clf.fit(X, y)

yp = clf.predict(XTest)

with open('dm-final-testpred.txt', 'w') as file:
    # print(indexTest.shape)
    # print(XTest.shape)
    # print(yp.shape)
    # data = np.concatenate((indexTest, XTest), axis=1)
    # data = np.concatenate((data, yp), axis=1)
    # print(data)   
    data = np.hstack((test, yp))
    # data = np.hstack((data, yp))
    df = pd.DataFrame(data, columns=range(0, 121))
    df.to_csv(file)
    file.close()

# plt.plot(np.hstack((XTest,yp))[0])
# plt.plot(XTest[0], c='r')
# # plt.plot([n,1:], c='g')
# plt.grid()
# plt.show()