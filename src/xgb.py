import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from generate_submission import generate_submission
# from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import Normalizer

X_train_raw = np.load("data5/x_train.npy")
Y_train = np.load("data5/y_train.npy")
X_valid_raw = np.load("data5/x_valid.npy")
Y_valid = np.load("data5/y_valid.npy")
X_test_raw = np.load("data5/x_test.npy")

# Pack into 500 length arrays of histone modification signals
X_train = [e.ravel() for e in X_train_raw]
X_valid = [e.ravel() for e in X_valid_raw]
X_test = [e.ravel() for e in X_test_raw]

X_full = np.concatenate([X_train, X_valid, X_test])

nm = Normalizer()
nm.fit(X_full)
X_train = nm.transform(X_train)
X_valid = nm.transform(X_valid)

#
# pca = TruncatedSVD(n_components=50)
# pca.fit(X_train)
# X_train = pca.transform(X_train)
# X_valid = pca.transform(X_valid)

dtrain = xgb.DMatrix(X_train, label=Y_train)
dvalid = xgb.DMatrix(X_valid, label=Y_valid)

params = {
    "silent": 1,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.05,
    "max_depth": 6,
}

before = datetime.now()

clf = xgb.train(params, dtrain,
                evals=[(dtrain, 'train'), (dvalid, 'valid')],
                # learning_rates=learning_rates,
                num_boost_round=241)

clf.save_model("weights/xgb/model.bin")