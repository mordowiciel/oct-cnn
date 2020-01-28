import logging
import sys

import wandb


class OCTLogger:
    def __init__(self, cfg, run_timestamp):
        self.cfg = cfg
        self.run_timestamp = run_timestamp

        self.__setup_wandb()
        self.__setup_python_logger()

    def __setup_wandb(self):
        wandb_run_name = '{}-{}-{}-{}'.format(self.cfg.network.architecture,
                                              self.run_timestamp,
                                              self.cfg.network.loss_function,
                                              self.cfg.network.optimizer)
        wandb.init(name=wandb_run_name)

    def __setup_python_logger(self):
        log = logging.getLogger('oct-cnn')
        log.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s - %(message)s')
        if self.cfg.dataset.training_dataset_path == '..\\dataset\\full\\train':
            file_handler = logging.FileHandler('{}/{}-FULL-{}-{}-{}.log'.format(self.cfg.misc.logs_path,
                                                                                self.cfg.network.architecture,
                                                                                self.run_timestamp,
                                                                                self.cfg.network.loss_function,
                                                                                self.cfg.network.optimizer))
            if self.cfg.dataset_config.generate_extended_test_dataset:
                file_handler = logging.FileHandler(
                    '{}/{}-FULL-EXTENDED-{}-{}-{}.log'.format(self.cfg.misc.logs_path,
                                                              self.cfg.network.architecture,
                                                              self.run_timestamp,
                                                              self.cfg.network.loss_function,
                                                              self.cfg.network.optimizer))
        else:
            file_handler = logging.FileHandler('{}/{}-{}-{}-{}.log'.format(self.cfg.misc.logs_path,
                                                                           self.cfg.network.architecture,
                                                                           self.run_timestamp,
                                                                           self.cfg.network.loss_function,
                                                                           self.cfg.network.optimizer))

        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        log.addHandler(file_handler)
        log.addHandler(stream_handler)
        return log

    def print_cfg(self):
        log = logging.getLogger('oct-cnn')
        log.info('Starting CNN training.')

        log.info('Using config path %s' % self.cfg.config_file_path)

        print()
        log.info('##### DATASET #####')
        log.info('Img size: %s', self.cfg.dataset.img_size)
        log.info('Input shape: %s', self.cfg.dataset.input_shape)
        log.info('Training dataset path: %s', self.cfg.dataset.training_dataset_path)
        log.info('Test dataset path: %s', self.cfg.dataset.test_dataset_path)
        if self.cfg.dataset.generate_extended_test_dataset:
            log.warning('USING EXTENDED TEST DATA GENERATOR')

        print()
        log.info('##### TRAINING #####')
        log.info('Epochs: %s', self.cfg.training.epochs)
        log.info('Training batch size: %s', self.cfg.training.training_batch_size)
        log.info('Test batch size: %s', self.cfg.training.test_batch_size)
        log.info('Early stopping monitor: %s', self.cfg.training.early_stopping_monitor)
        log.info('Early stopping patience: %s', self.cfg.training.early_stopping_patience)
        log.info('Early stopping min delta: %s', self.cfg.training.early_stopping_min_delta)

        print()
        log.info('##### AUGMENTATION #####')
        if self.cfg.augmentation.use_data_augmentation:
            log.warning('USING DATA AUGMENTATION')
            log.info('Horizontal flip: %s', self.cfg.augmentation.horizontal_flip)
            log.info('Width shift range: %s', self.cfg.augmentation.width_shift_range)
            log.info('Height shift range: %s', self.cfg.augmentation.height_shift_range)
            log.info('Brightness range: %s', self.cfg.augmentation.brightness_range)
            log.info('dtype: %s', self.cfg.augmentation.dtype)
        else:
            log.info("Skipping data augmentation.")

        print()
        log.info('##### NETWORK #####')
        log.info('Architecture: %s', self.cfg.network.architecture)
        log.info('Loss function: %s', self.cfg.network.loss_function)
        log.info('Optimizer: %s', self.cfg.network.optimizer)

        print()
        log.info('##### MISC #####')
        log.info('Model save path: %s', self.cfg.misc.models_path)
        log.info('Logs save path: %s', self.cfg.misc.logs_path)

# def __setup_python_logger(cfg, TIMESTAMP):
#     log = logging.getLogger('oct-cnn')
#     log.setLevel(logging.INFO)
#
#     formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s - %(message)s')
#     if cfg.dataset.training_dataset_path == '..\\dataset\\full\\train':
#         file_handler = logging.FileHandler('{}/{}-FULL-{}-{}-{}.log'.format(cfg.misc.logs_path,
#                                                                             cfg.network.architecture,
#                                                                             TIMESTAMP,
#                                                                             cfg.network.loss_function,
#                                                                             cfg.network.optimizer))
#         if cfg.dataset_config.generate_extended_test_dataset:
#             file_handler = logging.FileHandler(
#                 '{}/{}-FULL-EXTENDED-{}-{}-{}.log'.format(cfg.misc.logs_path,
#                                                           cfg.network.architecture, TIMESTAMP,
#                                                           cfg.network.loss_function,
#                                                           cfg.network.optimizer))
#     else:
#         file_handler = logging.FileHandler('{}/{}-{}-{}-{}.log'.format(cfg.misc.logs_path,
#                                                                        cfg.network.architecture,
#                                                                        TIMESTAMP,
#                                                                        cfg.network.loss_function,
#                                                                        cfg.network.optimizer))
#
#     file_handler.setFormatter(formatter)
#
#     stream_handler = logging.StreamHandler(stream=sys.stdout)
#     stream_handler.setLevel(logging.INFO)
#     stream_handler.setFormatter(formatter)
#
#     log.addHandler(file_handler)
#     log.addHandler(stream_handler)
#     return log
#
#
#
#
# def print_cfg(cfg):
#     log = logging.getLogger('oct-cnn')
#     log.info('Starting CNN training.')
#
#     print()
#     log.info('##### DATASET #####')
#     log.info('Img size: %s', cfg.dataset.img_size)
#     log.info('Input shape: %s', cfg.dataset.input_shape)
#     log.info('Training dataset path: %s', cfg.dataset.training_dataset_path)
#     log.info('Test dataset path: %s', cfg.dataset.test_dataset_path)
#     if cfg.dataset.generate_extended_test_dataset:
#         log.warning('USING EXTENDED TEST DATA GENERATOR')
#
#     print()
#     log.info('##### TRAINING #####')
#     log.info('Epochs: %s', cfg.training.epochs)
#     log.info('Training batch size: %s', cfg.training.training_batch_size)
#     log.info('Test batch size: %s', cfg.training.test_batch_size)
#
#     print()
#     log.info('##### AUGMENTATION #####')
#     if cfg.augmentation.use_data_augmentation:
#         log.warning('USING DATA AUGMENTATION')
#         log.info('Horizontal flip: %s', cfg.augmentation.horizontal_flip)
#         log.info('Width shift range: %s', cfg.augmentation.width_shift_range)
#         log.info('Height shift range: %s', cfg.augmentation.height_shift_range)
#         log.info('Brightness range: %s', cfg.augmentation.brightness_range)
#         log.info('dtype: %s', cfg.augmentation.dtype)
#     else:
#         log.info("Skipping data augmentation.")
#
#     print()
#     log.info('##### NETWORK #####')
#     log.info('Architecture: %s', cfg.network.architecture)
#     log.info('Loss function: %s', cfg.network.loss_function)
#     log.info('Optimizer: %s', cfg.network.optimizer)
#
#     print()
#     log.info('##### MISC #####')
#     log.info('Model save path: %s', cfg.misc.models_path)
#     log.info('Logs save path: %s', cfg.misc.logs_path)
