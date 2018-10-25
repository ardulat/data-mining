import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import ndimage
from scipy.signal import convolve2d

# barbara
image1 = ndimage.imread('barbara.png', flatten=True)

h1 = np.array([[1, -1], [-1, 1]])
h2 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
h3 = np.array([[1, 1, 1, 1, 1], [-1, -1, -1, -1, -1]])
h4 = h3.transpose()
h5 = np.ones((10, 10))

result1 = convolve2d(image1, h1, mode='same')
result2 = convolve2d(image1, h2, mode='same')
result3 = convolve2d(image1, h3, mode='same')
result4 = convolve2d(image1, h4, mode='same')
result5 = convolve2d(image1, h5, mode='same')

plt.figure(1)
plt.imshow(result1, interpolation='nearest', cmap='gray')
plt.figure(2)
plt.imshow(result2, interpolation='nearest', cmap='gray')
plt.figure(3)
plt.imshow(result3, interpolation='nearest', cmap='gray')
plt.figure(4)
plt.imshow(result4, interpolation='nearest', cmap='gray')
plt.figure(5)
plt.imshow(result5, interpolation='nearest', cmap='gray')
plt.show()

# phantom
image2 = ndimage.imread('SheppLoganPhantom.png', flatten=True)

result6 = convolve2d(image2, h1, mode='valid')
result7 = convolve2d(image2, h2, mode='valid')
result8 = convolve2d(image2, h3, mode='valid')
result9 = convolve2d(image2, h4, mode='valid')
result10 = convolve2d(image2, h5, mode='valid')

plt.figure(6)
plt.imshow(result6, interpolation='nearest', cmap='gray')
plt.figure(7)
plt.imshow(result7, interpolation='nearest', cmap='gray')
plt.figure(8)
plt.imshow(result8, interpolation='nearest', cmap='gray')
plt.figure(9)
plt.imshow(result9, interpolation='nearest', cmap='gray')
plt.figure(10)
plt.imshow(result10, interpolation='nearest', cmap='gray')
plt.show()