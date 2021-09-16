from tensorflow import keras
import numpy as np


class ConsecutiveEarlyStopping(keras.callbacks.Callback):
    """Stop training when 'patience' epochs are consecutively worse

    Arguments:
        patience: Number of epochs to wait after which we will terminate
    """

    def __init__(self, model_path='cnn', monitor='val_loss', min_delta=0, patience=0, mode='max', restore_best_weights=False, save_steps=True):
        super(ConsecutiveEarlyStopping, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor
        self.restore_best_weights = restore_best_weights
        self.count = 0
        self.prev = None
        self.model_path = model_path
        self.save_steps = save_steps

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        better = False
        if self.save_steps == True:
            self.model.save(self.model_path)

        if self.prev == None:
            better = True
        elif self.mode == 'min':
            better = np.less(current, self.prev)
        elif self.mode == 'max':
            better = np.greater(current, self.prev)

        if better:
            self.prev = current
            self.count = 0
            self.best_weights = self.model.get_weights()
        else:
            self.count += 1
            self.prev = current
            if self.count > self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    self.model.set_weights(self.best_weights)
