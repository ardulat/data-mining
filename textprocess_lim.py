#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import decomposition


corpus = ['To be, or not to be, that is the question',
           'Whether tis nobler in the mind to suffer',
           'The slings and arrows of outrageous fortune',
           'Or to take arms against a sea of troubles',
           'And by doing something',
           'the the the the the the the'
]

# corpus = [
#     'The dog barked.',
#     'I like chewing gum.',
#     'The cat meowed.'
# ]

vectorizer = CountVectorizer(min_df=1)

dt = vectorizer.fit_transform(corpus)

x = vectorizer.get_feature_names()

a = dt.toarray()

original = a

print('BEFORE APPLYING SVD:\n')
print(a)

transformer = TfidfTransformer(smooth_idf=False)

tfidf = transformer.fit_transform(a).toarray()

u,s,v = np.linalg.svd(tfidf, full_matrices=False)

a = np.dot(u, np.dot(np.diag(s), v))

approximated = a

print('\nAFTER APPLYING SVD:\n')
print(a)

for i in range(len(original)):
    similarity = np.dot(original[i], approximated[i]) / (np.linalg.norm(original[i])*np.linalg.norm(approximated[i]))
    print ("Similarity for row %d is: %f" % (i, similarity))

