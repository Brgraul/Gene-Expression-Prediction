import numpy as np
from keras.models import load_model
from generate_submission import generate_submission
from models.deepchrome import prepare_X

X_train_raw = np.load("data10/x_train.npy")
X_valid_raw = np.load("data10/x_valid.npy")
X_test_raw = np.load("data10/x_test.npy")

X_full_raw = np.concatenate([X_train_raw, X_valid_raw, X_test_raw])

X_test = prepare_X(X_test_raw, X_full_raw)

model = load_model("weights/deepchrome/0.9256-0.8642-2017-03-01 18:14:13.479762-08.hdf5")

model.summary()

predictions = model.predict_proba(X_test)
generate_submission(predictions[:,1].ravel())
