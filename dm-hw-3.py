import numpy as np

x = np.array([1,1,1,0,0,0])
y = np.array([0,0,0,1,1,1])
print (np.cov(x, y))