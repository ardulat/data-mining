from sklearn import datasets
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt


digits = datasets.load_digits()
A = digits.data

for n in range(1,65):
    pca = PCA(n_components=n)

    B = pca.fit_transform(A)

    ADR = pca.inverse_transform(B)

    X = ADR[110,:]
    y = A[110,:]

    fig = plt.figure()
    fig1 = fig.add_subplot(1, 2, 1)
    fig1.imshow(A[110,:].reshape(8,8), interpolation='none', cmap='gray')
    plt.title('Original')
    fig2 = fig.add_subplot(1, 2, 2)
    fig2.imshow(X.reshape(8,8), interpolation='none', cmap='gray')
    plt.title('n_component = %d' % n)
    plt.show()