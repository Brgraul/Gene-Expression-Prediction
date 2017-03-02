import numpy as np
from keras.models import load_model, Sequential
from keras.layers import Dense
from generate_submission import generate_submission
from models.deepchrome import prepare_X, prepare_Y
from sklearn.preprocessing import Normalizer
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from roc_auc_progbar_logger import ROCAUCProgbarLogger
from sklearn.svm import SVC

X_train_raw = np.load("data/x_train.npy")
Y_train_raw = np.load("data/y_train.npy")
X_valid_raw = np.load("data/x_valid.npy")
Y_valid_raw = np.load("data/y_valid.npy")
X_test_raw = np.load("data/x_test.npy")

X_full_raw = np.concatenate([X_train_raw, X_valid_raw, X_test_raw])

X_train_dc = prepare_X(X_train_raw, X_full_raw)
Y_train_dc = prepare_Y(Y_train_raw)
X_valid_dc = prepare_X(X_valid_raw, X_full_raw)
Y_valid_dc = prepare_Y(Y_valid_raw)
X_test_dc = prepare_X(X_test_raw, X_full_raw)

model_dc = load_model("weights/deepchrome/0.9160-0.8370-2017-02-22 21:33:26.787743-15.hdf5")

X_train_raw_xgb = np.array([e.ravel() for e in X_train_raw])
X_valid_raw_xgb = np.array([e.ravel() for e in X_valid_raw])
X_test_raw_xgb = np.array([e.ravel() for e in X_test_raw])

X_full_raw_xgb = np.concatenate([X_train_raw_xgb, X_valid_raw_xgb, X_test_raw_xgb])

nm = Normalizer()
nm.fit(X_full_raw_xgb)
X_train_xgb = nm.transform(X_train_raw_xgb)
X_valid_xgb = nm.transform(X_valid_raw_xgb)
X_test_xgb = nm.transform(X_test_raw_xgb)

dtrain = xgb.DMatrix(X_train_xgb, label=Y_train_raw)
dvalid = xgb.DMatrix(X_valid_xgb, label=Y_valid_raw)
dtest = xgb.DMatrix(X_test_xgb)

model_xgb = xgb.Booster()
model_xgb.load_model("weights/xgb/model.bin")

Y_train_pred_xgb = model_xgb.predict(dtrain)
Y_valid_pred_xgb = model_xgb.predict(dvalid)
Y_test_pred_xgb = model_xgb.predict(dtest)

Y_train_pred_dc = model_dc.predict_proba(X_train_dc)[:, 1].ravel()
Y_valid_pred_dc = model_dc.predict_proba(X_valid_dc)[:, 1].ravel()
Y_test_pred_dc = model_dc.predict_proba(X_test_dc)[:, 1].ravel()

X_train_stack = []
for i in range(len(X_train_xgb)):
    x = np.concatenate([X_train_xgb[i], [Y_train_pred_xgb[i], Y_train_pred_dc[i]]])
    X_train_stack.append(x)
X_train_stack = np.array(X_train_stack)

X_valid_stack = []
for i in range(len(X_valid_xgb)):
    x = np.concatenate([X_valid_xgb[i], [Y_valid_pred_xgb[i], Y_valid_pred_dc[i]]])
    X_valid_stack.append(x)
X_valid_stack = np.array(X_valid_stack)

X_test_stack = []
for i in range(len(X_test_xgb)):
    x = np.concatenate([X_test_xgb[i], [Y_test_pred_xgb[i], Y_test_pred_dc[i]]])
    X_test_stack.append(x)
X_test_stack = np.array(X_test_stack)


# clf = LogisticRegression(C=10.0 ** -3, penalty="l1")
# clf.fit(X_train_stack, Y_train_raw)
# Y_valid_pred_stack = clf.predict_proba(X_valid_stack)[:, 1].ravel()

clf = SVC(kernel="sigmoid", probability=True)
clf.fit(X_train_stack, Y_train_raw)
Y_valid_pred_stack = clf.predict_proba(X_valid_stack)[:, 1].ravel()

# stack = Sequential()
# stack.add(Dense(625, input_shape=(502,), activation="relu"))
# stack.add(Dense(125, activation="relu"))
# stack.add(Dense(2, activation="softmax"))
#
# # opti = SGD(lr=1e-3, decay=1e-7, momentum=0, nesterov=True)
# # opti = "sgd"
# opti = "adadelta"
# # opti = "adagrad"
# # opti = Adagrad(lr=1e-3, decay=1e-7)
# # opti = "rmsprop"
# stack.compile(loss="binary_crossentropy",
#               optimizer=opti,
#               metrics=["accuracy"])
#
# stack.summary()
#
# stack.fit(X_train_stack, Y_train_dc, batch_size=32,
#                                      nb_epoch=1000,
#                                      validation_data=(X_valid_stack, Y_valid_dc),
#                                      verbose=0,
#                                      callbacks=[ROCAUCProgbarLogger(verbose=1)])


score1 = roc_auc_score(Y_valid_raw, Y_valid_pred_xgb)
score2 = roc_auc_score(Y_valid_raw, Y_valid_pred_dc)
score3 = roc_auc_score(Y_valid_raw, Y_valid_pred_stack)

print score1, score2, score3

# generate_submission(predictions)