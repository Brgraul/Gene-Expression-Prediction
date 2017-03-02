# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import StratifiedKFold

x_train_raw = np.load("raw_data/x_train.npy")
x_test_raw = np.load("raw_data/x_test.npy")
y_train_raw = np.load("raw_data/y_train.npy")

X = np.array(np.split(x_train_raw[:, 1:], x_train_raw.shape[0] / 100))
X_test = np.array(np.split(x_test_raw[:, 1:], x_test_raw.shape[0] / 100))
Y = y_train_raw[:, 1]

models = []
kf = StratifiedKFold(n_splits=2,
                     shuffle=True)
train_index, valid_index = next(iter(kf.split(X, Y)))

X_train, X_valid = X[train_index], X[valid_index]
Y_train, Y_valid = Y[train_index], Y[valid_index]

np.save("data/x_train.npy", X_train)
np.save("data/y_train.npy", Y_train)
np.save("data/x_valid.npy", X_valid)
np.save("data/y_valid.npy", Y_valid)
np.save("data/x_test.npy", X_test)
