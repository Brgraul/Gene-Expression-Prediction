import numpy as np
from keras.models import load_model
from generate_submission import generate_submission
from models.deepchrome import prepare_X

x_test_raw = np.load("data/x_test.npy")

X_test = prepare_X(x_test_raw)

model = load_model("weights/0.9106-0.3888-2017-02-13 21:27:30.334334-38.hdf5")

model.summary()

predictions = model.predict_proba(X_test)
generate_submission(predictions[:,1].ravel())
