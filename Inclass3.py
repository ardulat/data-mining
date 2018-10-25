import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import linear_model


tt = pd.read_csv("lasso-example-data.txt", header=None)

X=tt.values[:,0:4]
x=tt.values[:,1]
y=tt.values[:,5]

alpha_list = np.linspace(10, 0.0001, num=10000)
coef_list = []

for alpha in alpha_list:
    lasmodel = linear_model.Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    lasmodel.fit(X,y)
    coef_list.append(lasmodel.coef_)

plt.plot(alpha_list, coef_list)
plt.title("Alpha vs. Coeficients")
plt.xscale('log')
plt.show()