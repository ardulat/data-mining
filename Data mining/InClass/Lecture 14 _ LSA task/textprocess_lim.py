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

vectorizer = CountVectorizer(min_df=1)

dt = vectorizer.fit_transform(corpus)

x = vectorizer.get_feature_names()

dt2 = vectorizer.fit_transform(corpus)
print(dt2.toarray())


print(a)
