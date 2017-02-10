# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

# valid gets much better results

model = Sequential()
model.add(Convolution2D(50, 5, 10,
                        input_shape=(5, 100, 1),
                        activation='relu',
                        border_mode='valid'))
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(625, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(125, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
