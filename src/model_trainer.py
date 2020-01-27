import logging
from glob import glob

from callbacks.batch_history_callback import BatchHistory
from callbacks.time_history_callback import TimeHistory
from oct_utils.plot_utils import save_mse_to_epoch_graph, save_loss_to_batch_graph, save_accuracy_to_epoch_graph

log = logging.getLogger('oct-cnn')


class ModelTrainer:
    def __init__(self, cfg, model, training_data_generator, validation_data_generator, run_timestamp):
        self.cfg = cfg
        self.model = model
        self.training_data_generator = training_data_generator
        self.validation_data_generator = validation_data_generator
        self.run_timestamp = run_timestamp

    def __count_images(self, dir_path):
        return len(glob('{}//**//*.jpeg'.format(dir_path), recursive=True))

    def __save_model(self):
        model_path = '{}//{}-{}-{}-{}.h5'.format(self.cfg.misc.models_path,
                                                 self.cfg.network.architecture,
                                                 self.run_timestamp,
                                                 self.cfg.network.loss_function,
                                                 self.cfg.network.optimizer)
        log.info('Saving model at path %s', model_path)
        self.model.save(model_path)

    def train_model(self):
        log.info('Fitting model...')
        batch_history = BatchHistory(granularity=100)
        time_history = TimeHistory()

        training_dataset_image_count = self.__count_images(self.cfg.dataset.training_dataset_path)
        training_dataset_split = 1 - self.cfg.dataset.validation_split

        training_steps_per_epoch = int(
            training_dataset_image_count * training_dataset_split) // self.cfg.training.training_batch_size
        log.info('Training steps per epoch: %s', training_steps_per_epoch)

        val_steps_per_epoch = int(
            training_dataset_image_count * self.cfg.dataset.validation_split) // self.cfg.training.training_batch_size
        log.info('Validation steps per epoch: %s', val_steps_per_epoch)

        history = self.model.fit_generator(generator=self.training_data_generator,
                                           use_multiprocessing=False,
                                           validation_data=self.validation_data_generator,
                                           validation_steps=val_steps_per_epoch,
                                           epochs=self.cfg.training.epochs,
                                           callbacks=[batch_history, time_history],
                                           steps_per_epoch=training_steps_per_epoch)

        log.info('Model training complete.')
        log.info('Epoch training times: %s', time_history.epochs_training_duration)

        log.info('Saving MSE to epoch graph.')
        save_mse_to_epoch_graph(history, self.cfg.misc.logs_path)

        log.info('Saving accuracy to epoch graph.')
        save_accuracy_to_epoch_graph(history, self.cfg.misc.logs_path)

        log.info('Saving loss to batch graph')
        save_loss_to_batch_graph(batch_history.history['batch'],
                                 batch_history.history['loss'],
                                 self.cfg.misc.logs_path)

        self.__save_model()
        return self.model
