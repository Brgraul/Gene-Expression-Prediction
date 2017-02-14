# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution1D, MaxPooling1D
from keras.optimizers import SGD
import numpy as np

def prepare_X(X_raw):
    return np.array(np.array_split(X_raw[:, 1:], X_raw.shape[0] / 100))

def prepare_Y(Y_raw):
    return np.array([np.array([1, 0] if y == 0 else [0, 1]) for y in Y_raw[:, 1:].ravel()])

model = Sequential()

model.add(Convolution1D(nb_filter=87,
                        filter_length=10,
                        input_shape=(100, 5),
                        activation="relu",
                        border_mode="valid"))

model.add(MaxPooling1D(pool_length=5,
                       stride=5))

model.add(Flatten())

model.add(Dropout(0.85))

model.add(Dense(625, activation="relu"))

# model.add(Dropout(0.5))

model.add(Dense(125, activation="relu"))

# model.add(Dropout(0.5))

model.add(Dense(2, activation="softmax"))
