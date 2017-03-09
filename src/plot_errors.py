import numpy as np
from keras.models import load_model
from models.deepchrome import prepare_X, prepare_Y
import matplotlib.pyplot as plt

X_valid_raw = np.load("../data/x_valid.npy")
Y_valid_raw = np.load("../data/y_valid.npy")

X_valid = prepare_X(X_valid_raw, X_valid_raw)
Y_valid = prepare_Y(Y_valid_raw)

model = load_model("../weights/old2/0.9159-0.8538-2017-02-22 21:33:26.787743-11.hdf5")

model.summary()

Y_pred = model.predict_proba(X_valid)
wrong = []
for i, ((pred0, pred1), (valid0, valid1)) in enumerate(zip(Y_pred, Y_valid)):
    if (valid1 > valid0) != (pred1 > pred0):
        wrong.append((pred1, valid1, X_valid[i]))

print len(wrong), len(Y_pred)
#
# for i, (pred1, valid1, x) in enumerate(wrong):
#     plt.figure(i)
#     plt.title("%f: %f" % (pred1, valid1))
#     H3K4me3, = plt.plot(x[:,0].ravel(), label="H3K4me3")
#     H3K4me1, = plt.plot(x[:,1].ravel(), label="H3K4me1")
#     H3K36me3, = plt.plot(x[:,2].ravel(), label="H3K36me3")
#     H3K9me3, = plt.plot(x[:,3].ravel(), label="H3K9me3")
#     H3K27me3, = plt.plot(x[:,4].ravel(), label="H3K27me3")
#     plt.legend(handles=[H3K4me3, H3K4me1, H3K36me3, H3K9me3, H3K27me3])
