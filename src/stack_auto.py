# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
from sklearn.metrics import roc_auc_score
from itertools import product

from roc_auc_progbar_logger import ROCAUCProgbarLogger
from models.deepchrome import create_model, prepare_X, prepare_Y

X_train_raw = np.load("data/x_train.npy")
Y_train_raw = np.load("data/y_train.npy")
X_valid_raw = np.load("data/x_valid.npy")
Y_valid_raw = np.load("data/y_valid.npy")
X_test_raw = np.load("data/x_test.npy")
X_full_raw = np.concatenate([X_train_raw, X_valid_raw, X_test_raw])

X_train = prepare_X(X_train_raw, X_full_raw)
Y_train = prepare_Y(Y_train_raw)
X_valid = prepare_X(X_valid_raw, X_full_raw)
Y_valid = prepare_Y(Y_valid_raw)
X_test = prepare_X(X_test_raw, X_full_raw)

name = datetime.now().strftime("%Y-%m-%d-%H-%M")

Y_train_preds = None
Y_valid_preds = None
Y_test_preds = None

possibilities = product([40, 50, 60], [10, 11], [5, 6])
for i, (nb_filter, filter_length, pool_length) in enumerate(possibilities):
    model = create_model(nb_filter=nb_filter,
                         filter_length=filter_length,
                         pool_length=pool_length)

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    model.summary()

    path = "weights/stack/%s-%i.hdf5" % (name, i)

    callbacks = [
        ROCAUCProgbarLogger(verbose=1),
        ModelCheckpoint(path,
                        monitor="val_loss",
                        save_best_only=True),
        EarlyStopping(monitor="val_loss",
                      patience=25,
                      mode="min"),
    ]

    model.fit(X_train, Y_train, batch_size=32,
                                nb_epoch=10000,
                                validation_data=(X_valid, Y_valid),
                                verbose=0,
                                callbacks=callbacks)

    model.load_weights(path)

    Y_train_pred = model.predict_proba(X_train)[:, 1].ravel()
    Y_valid_pred = model.predict_proba(X_valid)[:, 1].ravel()
    Y_test_pred = model.predict_proba(X_test)[:, 1].ravel()

    if Y_train_preds is None:
        Y_train_preds = np.array([np.array(Y_train_pred)])
    else:
        Y_train_preds = np.concatenate([Y_train_preds, np.array([Y_train_pred])])

    if Y_valid_preds is None:
        Y_valid_preds = np.array([np.array(Y_valid_pred)])
    else:
        Y_valid_preds = np.concatenate([Y_valid_preds, np.array([Y_valid_pred])])

    if Y_test_preds is None:
        Y_test_preds = np.array([np.array(Y_test_pred)])
    else:
        Y_test_preds = np.concatenate([Y_test_preds, np.array([Y_test_pred])])

    print "AUC: %.6f" % roc_auc_score(Y_valid_raw, Y_valid_pred)

    Y_valid_pred_mean = np.mean(Y_valid_preds, axis=0)
    print "Mean AUC: %.6f" % roc_auc_score(Y_valid_raw, Y_valid_pred_mean)

np.save("data/y_train_preds.npy", Y_train_preds)
np.save("data/y_valid_preds.npy", Y_valid_preds)
np.save("data/y_test_preds.npy", Y_test_preds)