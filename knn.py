# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score

print "Loading data"
x_train_raw = np.load("data/x_train.npy")
y_train_raw = np.load("data/y_train.npy")
x_test_raw = np.load("data/x_test.npy")

x_train = np.split(x_train_raw[:, 1:], x_train_raw.shape[0] / 100)
y_train = y_train_raw[:, 1:].ravel()
x_test = np.split(x_test_raw[:, 1:], x_test_raw.shape[0] / 100)

x = [e.T.ravel() for e in x_train]
x2 = [e.T.ravel() for e in x_test]
y = y_train

def sq_dist(x1, x2):
    return np.linalg.norm(x1 - x2)

# Accuracy: 0.645 (+/- 0.080)
# Accuracy: 0.623 (+/- 0.081)
# Accuracy: 0.658 (+/- 0.076)
# Accuracy: 0.767 (+/- 0.081)
# Accuracy: 0.485 (+/- 0.081)

weights = [0, 0.2, 0, 1, 0]
    
def metric(x1, x2):
    if np.shape(x1) == (10,):
        return 0.0
    x1a, x1b, x1c, x1d, x1e = np.split(x1, 5)
    x2a, x2b, x2c, x2d, x2e = np.split(x2, 5)
    return weights[0] * sq_dist(x1a, x2a) + weights[1] * sq_dist(x1b, x2b) + weights[2] * sq_dist(x1c, x2c) + weights[3] * sq_dist(x1d, x2d) + weights[4] * sq_dist(x1e, x2e) 

x = x[:400]
y = y[:400]
    
clf = KNeighborsClassifier(n_neighbors=1, metric='pyfunc', metric_params={"func": metric})
#clf.fit(x, y)
scores = cross_val_score(clf, x, y, scoring="roc_auc")
print "Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2)