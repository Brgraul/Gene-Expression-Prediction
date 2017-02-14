from math import ceil

# Split without suffling.
def split_train_test(X, Y, test_size):
    test_idx = int(ceil(len(X) * (1 - test_size)))
    X_test = X[test_idx:]
    Y_test = Y[test_idx:]
    X_train = X[:test_idx]
    Y_train = Y[:test_idx]
    return (X_train, Y_train), (X_test, Y_test)
