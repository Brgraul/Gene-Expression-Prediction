# -*- coding: utf-8 -*-
from keras.callbacks import ModelCheckpoint
import numpy as np
import datetime

from models.deepchrome import model
# from models.custom import model

x_train_raw = np.load("data/x_train.npy")
y_train_raw = np.load("data/y_train.npy")

# shape: (n_samples, height=100, width=5)
X_train = np.array(np.split(x_train_raw[:, 1:], x_train_raw.shape[0] / 100))
# For custom model
# X_train = X_train[:,:,:,np.newaxis].transpose((0, 2, 1, 3))
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[2], X_train.shape[1], 1))
# [
#   [0, 1], -> activated
#   [1, 0], -> not activated
# ]
Y_train = np.array([np.array([1, 0] if y == 0 else [0, 1]) for y in y_train_raw[:, 1:].ravel()])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

nb_epochs = 100
batch_size = 32

filename = "weights/%s-{epoch:02d}-{val_loss:.4f}.hdf5" % str(datetime.datetime.now())
model_checkpoint = ModelCheckpoint(filename, monitor="val_loss",
                                             verbose=0,
                                             save_best_only=False,
                                             save_weights_only=False,
                                             mode='auto',
                                             period=1)

model.fit(X_train, Y_train, batch_size=batch_size,
                            nb_epoch=nb_epochs,
                            validation_split=0.2,
                            callbacks=[model_checkpoint])
