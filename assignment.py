import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error


tt = pd.read_csv('dm-final-train.txt', header=None)
test = pd.read_csv('dm-final-testdist.txt', header=None)

index = tt.values[:, 0]
# X_train = tt.values[:1500, 1:101]
# X_test = tt.values[1501:, 1:101]
# y_train = tt.values[:1500, 101:121]
# y_test = tt.values[1501:, 101:121]
X_train = tt.values[:, 1:101]
y_train = tt.values[:, 101:121]
indexTest = test.values[:, 0]
X_test = test.values[:, :]

clf = LinearRegression()
# clf.fit(X, y)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test[:,1:])
# yp = clf.predict(XTest[:,1:])

# mse = mean_squared_error(y_test, y_pred)
# print(math.sqrt(mse))

with open('dm-final-testpred.txt', 'w') as file:
    data = np.hstack((X_test, y_pred))
    df = pd.DataFrame(data, columns=range(len(data[0])))
    df.to_csv(file, index=False)
    file.close()

# plt.plot(np.hstack((X_test,y_test))[0])
plt.plot(np.hstack((X_test,y_pred))[0], c='r')
plt.plot(X_test[0])
# plt.plot([n,1:], c='g')
plt.grid()
plt.show()