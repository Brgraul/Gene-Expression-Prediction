# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:08:45 2017

@author: Antoine
"""

import numpy as np
import sklearn.neighbors
import sklearn.svm
from sklearn.cross_validation import train_test_split, cross_val_score
import sklearn.ensemble
from generate_submission import generate_submission

print ("Loading data")
x_train_raw = np.load("data/x_train.npy")
y_train_raw = np.load("data/y_train.npy")
x_test_raw = np.load("data/x_test.npy")

x_train = [e.ravel() for e in np.split(x_train_raw[:, 1:], x_train_raw.shape[0] / 100)]
y_train = y_train_raw[:, 1:].ravel()
x_test = [e.ravel() for e in np.split(x_test_raw[:, 1:], x_test_raw.shape[0] / 100)]


# print("Starting KNN")
# KNN = sklearn.neighbors.KNeighborsClassifier(algorithm = "ball_tree")
# KNN.fit(X = x_train, y = y_train)
# #res = KNN.predict(x_test)
# scores = cross_val_score(KNN, x_train, y_train, scoring="roc_auc")
# print ("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
#
#
#
# print("Starting SVM")
# SVM = sklearn.svm.SVC()
# SVM.fit(X = x_train, y = y_train)
# scores = cross_val_score(SVM, x_train, y_train, scoring="roc_auc")
# print ("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))


print("Starting RFC")
RFC = sklearn.ensemble.RandomForestClassifier(n_estimators = 10000, n_jobs=-1)
RFC.fit(X = x_train, y = y_train)
scores = cross_val_score(RFC, x_train, y_train, scoring="roc_auc")
print ("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

predictions = RFC.predict_proba(x_test)[:,1].ravel()

generate_submission(predictions)
