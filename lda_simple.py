import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


tt = pd.read_csv('simple_bin_classifier.txt', header=None)
X = tt.values[:, 0:2]
y = tt.values[:, 2]

#print(X)
I = y == 1
# list comprehension
J = [not x for x in I]

# selection with array of booleans
print("True:")
print(X[I,:])

print("False:")
print(X[J,:])


# computing the mean vectors
m1 = np.mean(X[I,:],axis=0)
m2 = np.mean(X[J,:],axis=0)

# find a linear classifier 'w' according
# to the LDA recipe from the slides
# Also, find the best value for the threshold

# Compute sample covariances in each class S1 and S2
cov1 = np.cov((X[I,:]).T)
cov2 = np.cov((X[J,:]).T)

#Compute the within class scatter matrix SW = S1 + S2
Sw = cov1 + cov2
# print(Sw)
w = np.dot(np.linalg.inv(Sw),(m1 - m2))

values = np.dot(w.T, X.T)

print (values)

threshold = np.mean(values)

for x in values:
    # threshold = 6
    if x > threshold:
        # everything more than threshold is true
        isTrue = 1
    else:
        # everything less than threshold is false
        isTrue = 0

fig, ax = plt.subplots()
ax.plot(X[:,0], (threshold - X[:,0]*w[0])/w[1], color = "red")

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')

plt.title('d')
plt.xlabel('X1')
plt.ylabel('X2')

plt.xticks(())
plt.yticks(())
plt.show()