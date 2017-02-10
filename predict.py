import numpy as np
from keras.models import load_model
from generate_submission import generate_submission

x_test_raw = np.load("data/x_test.npy")

# shape: (n_samples, height=100, width=5)
X_test = np.array(np.split(x_test_raw[:, 1:], x_test_raw.shape[0] / 100))
# For custom model
# X_test = X_test[:,:,:,np.newaxis].transpose((0, 2, 1, 3))

X_test = np.array(X_test)

model = load_model("weights/2017-02-10 11:12:42.421565-86-0.3702.hdf5")

model.summary()

predictions = model.predict_proba(X_test)
generate_submission(predictions[:,1].ravel())
