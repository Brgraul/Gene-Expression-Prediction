# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

print "Loading data"
x_train_raw = np.load("data/x_train.npy")
y_train_raw = np.load("data/y_train.npy")
x_test_raw = np.load("data/x_test.npy")

x_train = np.split(x_train_raw[:, 1:], x_train_raw.shape[0] / 100)
y_train = y_train_raw[:, 1:].ravel()
x_test = np.split(x_test_raw[:, 1:], x_test_raw.shape[0] / 100)

def plot_gene(i):
    print y_train[i]
    plt.plot(x_train[i])
