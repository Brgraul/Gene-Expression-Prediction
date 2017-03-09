import numpy as np
from keras.models import load_model
from models.deepchrome import prepare_X, prepare_Y
from sklearn.metrics import roc_auc_score
from generate_submission import generate_submission

X_train_raw = np.load("data/x_train.npy")
X_valid_raw = np.load("data/x_valid.npy")
X_test_raw = np.load("data/x_test.npy")
Y_valid_raw = np.load("data/y_valid.npy")

X_full_raw = np.concatenate([X_train_raw, X_valid_raw, X_test_raw])

X_valid = prepare_X(X_valid_raw, X_full_raw)
Y_valid = prepare_Y(Y_valid_raw)
X_test = prepare_X(X_test_raw, X_full_raw)

models = []

Y_preds = np.array([model.predict_proba(X_test) for model in models])

Y_pred = []
for i in range(len(X_test)):
    # best = np.argmax(np.abs(Y_preds[:, i, 1].ravel() - 0.5))
    # Y_pred.append(Y_preds[best, i, 1])
    mean = np.mean(Y_preds[:, i, 1])
    Y_pred.append(mean)

# print roc_auc_score(Y_valid[:, 1], Y_pred)
generate_submission(Y_pred)
