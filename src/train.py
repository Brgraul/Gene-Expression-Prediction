# -*- coding: utf-8 -*-
from keras.callbacks import ModelCheckpoint
import numpy as np
import datetime

from utils import split_train_test
from roc_auc_progbar_logger import ROCAUCProgbarLogger
from models.deepchrome import model, prepare_X, prepare_Y
# from models.custom import model

x_train_raw = np.load("data/x_train.npy")
y_train_raw = np.load("data/y_train.npy")

X_train = prepare_X(x_train_raw)
Y_train = prepare_Y(y_train_raw)

nb_epochs = 10000
batch_size = 32

model.summary()

# opti = SGD(lr=1e-3, decay=1e-7, momentum=0, nesterov=True)
# opti = "adadelta"
opti = "adagrad"
# opti = "rmsprop"
model.compile(loss="categorical_crossentropy",
              optimizer=opti,
              metrics=["accuracy"])

filename = "weights/{roc_auc:.4f}-{val_loss:.4f}-%s-{epoch:02d}.hdf5" % \
           str(datetime.datetime.now())
model_checkpoint = ModelCheckpoint(filename, monitor="val_loss",
                                             verbose=0,
                                             save_best_only=False,
                                             save_weights_only=False,
                                             mode="auto")

(X_train, Y_train), (X_test, Y_test) = split_train_test(X_train, Y_train, 0.2)

model.fit(X_train, Y_train, batch_size=batch_size,
                            nb_epoch=nb_epochs,
                            validation_data=(X_test, Y_test),
                            verbose=0,
                            callbacks=[ROCAUCProgbarLogger(), model_checkpoint])
