# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import itertools
from generate_submission import generate_submission

print "Loading data"
x_train_raw = np.load("data/x_train.npy")
y_train_raw = np.load("data/y_train.npy")
x_test_raw = np.load("data/x_test.npy")

# Pack into 500 length arrays of histone modification signals
x_train = [e.ravel() for e in np.split(x_train_raw[:, 1:], x_train_raw.shape[0] / 100)]
y_train = y_train_raw[:, 1:].ravel()
x_test = [e.ravel() for e in np.split(x_test_raw[:, 1:], x_test_raw.shape[0] / 100)]

# From assig3.py
clf = LogisticRegression(C=10.0 ** -2, penalty="l1")
clf.fit(x_train, y_train)
# Only take the probability of the gene being active.
predictions = clf.predict_proba(x_test)[:,1].ravel()

generate_submission(predictions)