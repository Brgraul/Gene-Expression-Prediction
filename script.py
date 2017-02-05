# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import itertools

print "Loading data"
x_train_raw = np.load("data/x_train.npy")
y_train_raw = np.load("data/y_train.npy")

# Pack into 500 length arrays of histone modification signals
# No idea why, but having a shape of n * 500 instead of (n * 100) * 5 improves
# the accuracy from ~0.71 to ~0.85 for LogisticRegression.
# With Deep Learning, we keep the (n * 100) * 5
x_train = [e.ravel() for e in np.split(x_train_raw[:, 1:], x_train_raw.shape[0] / 100)]
y_train = y_train_raw[:, 1:].ravel()

for C, penalty in itertools.product(10.0 ** np.arange(-6, 1), ["l1", "l2"]):
    print "Training model with C=%.2e, penalty=%s" % (C, penalty)
    clf = LogisticRegression(C=C, penalty=penalty)
    clf.fit(X_train, Y_train)
    scores = cross_val_score(clf, x_train, y_train, scoring="roc_auc")
    print "Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2)