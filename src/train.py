# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import time
import pickle

from roc_auc_progbar_logger import ROCAUCProgbarLogger
from models.deepchrome import create_model, prepare_X, prepare_Y

nb_epoch = 1000
batch_size = 256
patience = 10

X_train_raw = np.load("data5/x_train.npy")
Y_train_raw = np.load("data5/y_train.npy")
X_valid_raw = np.load("data5/x_valid.npy")
Y_valid_raw = np.load("data5/y_valid.npy")
X_test_raw = np.load("data5/x_test.npy")

X_full_raw = np.concatenate([X_train_raw, X_valid_raw, X_test_raw])
X_train_5 = prepare_X(X_train_raw, X_full_raw)
Y_train_5 = prepare_Y(Y_train_raw)
X_valid_5 = prepare_X(X_valid_raw, X_full_raw)
Y_valid_5 = prepare_Y(Y_valid_raw)
X_test = prepare_X(X_test_raw, X_full_raw)

X_full = np.concatenate([X_train_5, X_valid_5])
Y_full = np.concatenate([Y_train_5, Y_valid_5])

params_space = {
    "nb_filter": hp.quniform("filter_length", 10, 150, 10),
    "filter_length": hp.quniform("nb_filter", 2, 14, 1),
    "dense_layers": [
        hp.quniform("layer_1", 10, 1000, 10),
        hp.quniform("layer_2", 10, 1000, 10),
    ],
    "pool_length": hp.quniform("pool_length", 2, 8, 1),
    "dropouts": [
        hp.uniform("dropout_1", 0, 1),
        hp.uniform("dropout_2", 0, 1),
        hp.uniform("dropout_3", 0, 1),
    ],
}

kf = StratifiedKFold(n_splits=3,
                     shuffle=False)

splits = list(kf.split(X_full, Y_full[:, 1].ravel()))

def objective(params):
    name = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    start_time = time.time()

    print "Training %s" % name
    print params

    i = 0
    scores = []
    # for train_index, valid_index in splits:
    i += 1
    # X_train, X_valid = X_full[train_index], X_full[valid_index]
    # Y_train, Y_valid = Y_full[train_index], Y_full[valid_index]
    X_train, X_valid = X_train_5, X_valid_5
    Y_train, Y_valid = Y_train_5, Y_valid_5

    model = create_model(**params)

    opti = "adam"
    model.compile(loss="binary_crossentropy",
                  optimizer=opti,
                  metrics=["accuracy"])

    # model.summary()

    path = "weights/cv/%s-%i.hdf5" % (name, i)
    callbacks = [
        ROCAUCProgbarLogger(verbose=0),
        ModelCheckpoint(path,
                        monitor="val_roc_auc",
                        verbose=0,
                        save_best_only=True,
                        save_weights_only=False,
                        mode="max"),
        EarlyStopping(monitor="val_roc_auc",
                      patience=patience,
                      mode="max"),
    ]

    model.fit(X_train, Y_train, batch_size=batch_size,
                                nb_epoch=nb_epoch,
                                validation_data=(X_valid, Y_valid),
                                verbose=0,
                                callbacks=callbacks)

    model.load_weights(path)

    Y_valid_pred = model.predict_proba(X_valid, verbose=0)

    score = roc_auc_score(Y_valid[:, 1].ravel(), Y_valid_pred[:, 1].ravel())

    print "%i: %.6f" % (i, score)

    scores.append(score)

    auc_mean = np.mean(scores)
    print "AUC: %0.6f (+/- %0.6f)" % (auc_mean, np.std(scores) * 2)

    end_time = time.time()
    print end_time - start_time

    return {
        "loss": -auc_mean,
        "status": STATUS_OK,
        "time": end_time - start_time,
    }


trials = Trials()
best = fmin(objective,
            space=params_space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print 'best:', best

print 'trials:'
for trial in trials.trials:
    print trial

pickle.dump(trials, open("trials.bin", "wb"))