from keras.callbacks import ProgbarLogger
from sklearn.metrics import roc_auc_score

class ROCAUCProgbarLogger(ProgbarLogger):
    def on_train_begin(self, logs=None):
        ProgbarLogger.on_train_begin(self, logs)
        self.verbose = 1

    def on_epoch_end(self, epoch, logs=None):
        if 'roc_auc' not in self.params['metrics']:
            self.params['metrics'].append('roc_auc')
        X_test, Y_test, _, _ = self.model.validation_data
        pred = self.model.predict_proba(X_test, verbose=0)
        logs['roc_auc'] = roc_auc_score(Y_test[:,1], pred[:,1])
        ProgbarLogger.on_epoch_end(self, epoch, logs)
