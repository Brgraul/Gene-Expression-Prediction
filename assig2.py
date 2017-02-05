# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import itertools

print "Loading data"
x_train_raw = np.load("data/x_train.npy")
y_train_raw = np.load("data/y_train.npy")
x_test_raw = np.load("data/x_test.npy")

# Pack into 500 length arrays of histone modification signals
# No idea why, but having a shape of n * 500 instead of (n * 100) * 5 improves
# the accuracy from ~0.71 to ~0.85 for LogisticRegression.
# With Deep Learning, we keep the (n * 100) * 5
x_train = [e.ravel() for e in np.split(x_train_raw[:, 1:], x_train_raw.shape[0] / 100)]
y_train = y_train_raw[:, 1:].ravel()
x_test = [e.ravel() for e in np.split(x_test_raw[:, 1:], x_test_raw.shape[0] / 100)]

# From assig3.py
clf = LogisticRegression(C=10.0 ** -2, penalty="l1")
clf.fit(x_train, y_train)
# Only take the probability of the gene being active.
predictions = clf.predict_proba(x_test)[:,1].ravel()

fp = open("prediction.csv", "w")
fp.write("GeneId,Prediction\n")
for idx, prediction in enumerate(predictions):
    fp.write("%i,%f\n" % ((idx + 1), prediction))
fp.close()