import numpy as np
from sklearn.metrics import roc_auc_score
from roc_auc_progbar_logger import ROCAUCProgbarLogger
from keras.models import Sequential
from keras.layers import Dense
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from models.deepchrome import prepare_Y

Y_train_raw = np.load("data/y_train.npy")
Y_valid_raw = np.load("data/y_valid.npy")
Y_train = prepare_Y(Y_train_raw)
Y_valid = prepare_Y(Y_valid_raw)

Y_train_preds = np.load("data/y_train_preds.npy").T
Y_valid_preds = np.load("data/y_valid_preds.npy").T
Y_test_preds = np.load("data/y_test_preds.npy").T

clf = SVC(probability=True)
clf.fit(Y_train_preds, Y_train_raw)
Y_valid_pred_stack = clf.predict_proba(Y_valid_preds)[:, 1].ravel()

print roc_auc_score(Y_valid_raw, Y_valid_pred_stack)

# clf = LogisticRegression(C=10.0 ** -1, penalty="l1")
# clf.fit(Y_train_preds, Y_train_raw)
# Y_valid_pred_stack = clf.predict_proba(Y_valid_preds)[:, 1].ravel()
#
# dtrain = xgb.DMatrix(Y_train_preds, label=Y_train_raw)
# dvalid = xgb.DMatrix(Y_valid_preds, label=Y_valid_raw)
#
# params = {
#     "silent": 1,
#     "objective": "binary:logistic",
#     "eval_metric": "auc",
#     "eta": 0.005,
#     "max_depth": 1,
# }
#
# bst = xgb.train(params, dtrain,
#                 num_boost_round=100000,
#                 evals=[(dtrain, 'train'), (dvalid, 'valid')])

# stack = Sequential()
# stack.add(Dense(625, input_shape=(9,), activation="relu"))
# stack.add(Dense(125, activation="relu"))
# stack.add(Dense(2, activation="softmax"))
#
# stack.compile(loss="binary_crossentropy",
#               optimizer="adam",
#               metrics=["accuracy"])
#
# stack.summary()
#
# stack.fit(Y_train_preds, Y_train_raw, batch_size=32,
#                                       nb_epoch=1000,
#                                       validation_data=(Y_valid_preds, Y_valid_raw),
#                                       verbose=0,
#                                       callbacks=[ROCAUCProgbarLogger(verbose=1)])



# generate_submission(predictions)