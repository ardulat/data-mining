#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn import datasets, linear_model

digits = datasets.load_digits()

print(digits.target)

B = digits.data

pca = PCA(n_components=3)

C = pca.fit_transform(B)

B_est = pca.inverse_transform(C)



# plt.plot(B[0,:])

# imp = plt.imshow(B[331,:].reshape((8,8)))
plt.figure(1)
plt.imshow(B[0,:].reshape(8,8),interpolation='nearest')
plt.figure(2)
plt.imshow(B_est[0,:].reshape(8,8),interpolation='nearest')

plt.figure(3)
# plt.plot(C[:,0],C[:,1],'.')

for i in range(10):
   plt.plot(C[digits.target==i,0],C[digits.target==i,1],'.')
    
plt.show()