# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution1D, MaxPooling1D
from keras.optimizers import SGD

# -- input dimensions
N_out = 50
k = 10
m = 5

# -- a typical modern convolution network (conv+relu+pool)
# model = nn.Sequential()
model = Sequential()
# -- stage 1 : filter bank -> squashing -> Max pooling
# model:add(nn.TemporalConvolution(nfeats, nstates[1], filtsize))
# model:add(nn.ReLU())
model.add(Convolution1D(N_out, k,
                        input_shape=(100, 5),
                        activation="relu",
                        border_mode="valid"))
# model:add(nn.TemporalMaxPooling(poolsize))
model.add(MaxPooling1D(m))

# -- stage 2 : standard 2-layer neural network
# model:add(nn.View(math.ceil((width-filtsize)/poolsize)*nstates[1]))
model.add(Flatten())
# model:add(nn.Dropout(0.5))
# 0.9 instead of 0.5 gives much more stable results
model.add(Dropout(0.75))
# model:add(nn.Linear(math.ceil((width-filtsize)/poolsize)*nstates[1], nstates[2]))
# model:add(nn.ReLU())
model.add(Dense(625, activation="relu"))

# additionnal
model.add(Dropout(0.75))

# model:add(nn.Linear(nstates[2], nstates[3]))
# model:add(nn.ReLU())
model.add(Dense(125, activation="relu"))

# model:add(nn.Linear(nstates[3], noutputs))
model.add(Dense(2, activation='softmax'))

model.summary()

# opti = SGD(lr=1e-3, decay=1e-7, momentum=0, nesterov=True)
# opti = 'adadelta'
# adagrad makes the val_loss actually follow the loss
opti = 'adagrad'
model.compile(loss='categorical_crossentropy',
              optimizer=opti,
              metrics=['accuracy'])
