from keras.callbacks import ProgbarLogger
from sklearn.metrics import roc_auc_score

class ROCAUCProgbarLogger(ProgbarLogger):
    def __init__(self, verbose=1):
        ProgbarLogger.__init__(self)
        self.verbose2 = verbose

    def on_train_begin(self, logs=None):
        ProgbarLogger.on_train_begin(self, logs)
        self.verbose = self.verbose2

    def on_epoch_end(self, epoch, logs=None):
        if 'roc_auc' not in self.params['metrics']:
            # self.params['metrics'].append('roc_auc')
            self.params['metrics'].append('val_roc_auc')
        # X_train, Y_train, _, _ = self.model.training_data
        X_test = list(self.model.validation_data[0:-3])
        Y_test = self.model.validation_data[-3]
        # train_pred = self.model.predict_proba(X_train, verbose=0)
        val_pred = self.model.predict_proba(X_test, verbose=0)
        # logs['roc_auc'] = roc_auc_score(Y_train[:,1], train_pred[:,1])
        logs['val_roc_auc'] = roc_auc_score(Y_test[:,1], val_pred[:,1])
        ProgbarLogger.on_epoch_end(self, epoch, logs)
