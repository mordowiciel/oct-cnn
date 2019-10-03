import keras.callbacks


class BatchHistory(keras.callbacks.Callback):

    def __init__(self, granularity):
        super().__init__()
        self.granularity = granularity
        self.last_batch_index = 0
        self.last_batch_threshold = 0

    def on_train_begin(self, logs=None):
        self.batch = []
        self.history = {}

    def on_batch_end(self, batch_index, logs=None):
        if batch_index == 0 and self.last_batch_index != 0:
            self.last_batch_threshold = self.last_batch_index + 1

        batch_index = self.last_batch_threshold + batch_index
        if batch_index % self.granularity == 0:
            logs = logs or {}
            self.batch.append(batch_index)
            for k, v in logs.items():
                if k == 'batch':
                    self.history.setdefault(k, []).append(batch_index)
                else:
                    self.history.setdefault(k, []).append(v)

        self.last_batch_index = batch_index
