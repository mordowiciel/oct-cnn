import time

import keras


class TimeHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.epochs_training_duration = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.epochs_training_duration.append(time.time() - self.epoch_time_start)
