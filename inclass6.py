import scipy.fftpack as fftpack
import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix


tt = pd.read_csv('tones_noise.txt', header=None)

X = tt.values[:, 0:501]
y = tt.values[:, 501]

XF = np.abs(fftpack.fft(X))

# LDA
clf = LinearDiscriminantAnalysis()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
score = (tp+tn)/(tn+fp+fn+tp)

print ('Usual LDA accuracy score: %f' % score)

clf = LinearDiscriminantAnalysis()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(XF, y)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
score = (tp+tn)/(tn+fp+fn+tp)

print ('Transformed (Fourier) LDA accuracy score: %f' % score)

# Decision Tree
clf = DecisionTreeClassifier()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
score = (tp+tn)/(tn+fp+fn+tp)

print ('Usual Decision Tree accuracy score: %f' % score)

clf = DecisionTreeClassifier()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(XF, y)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
score = (tp+tn)/(tn+fp+fn+tp)

print ('Transformed (Fourier) Decision Tree accuracy score: %f' % score)