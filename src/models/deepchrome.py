# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Merge, Dropout, Flatten, Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler, MaxAbsScaler, MinMaxScaler

def prepare_X(X_raw, X_full_raw):
    X_before = np.array([e.ravel() for e in X_raw])
    X_full_before = np.array([e.ravel() for e in X_full_raw])
    n = Normalizer()
    n.fit(X_full_before)
    result = np.array([np.split(e, 100) for e in n.transform(X_before)])
    return result

def prepare_Y(Y_raw):
    return np.array([np.array([1, 0] if y == 0 else [0, 1]) for y in Y_raw])

def predict(model, X):
    return model.predict_proba(X)[:, 1].ravel()

def create_model(nb_filter=50,
                 filter_length=10,
                 pool_length=5,
                 dense_layers=[625, 125],
                 dropouts=[0.5]):
    model = Sequential()

    model.add(Convolution1D(nb_filter=int(nb_filter),
                            filter_length=int(filter_length),
                            input_shape=(100, 5),
                            activation="relu",
                            border_mode="valid"))

    model.add(MaxPooling1D(pool_length=int(pool_length),
                           stride=int(pool_length)))

    model.add(Flatten())

    if len(dropouts) > 0 and dropouts[0] > 0:
        model.add(Dropout(dropouts[0]))

    for idx, dense in enumerate(dense_layers):
        model.add(Dense(int(dense), activation="relu"))

        if len(dropouts) > idx + 1 and dropouts[idx + 1] > 0:
            model.add(Dropout(dropouts[idx + 1]))

    model.add(Dense(2, activation="softmax"))

    return model
