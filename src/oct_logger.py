import logging
import sys


def setup_logger(cfg, TIMESTAMP):
    log = logging.getLogger('oct-cnn')
    log.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s - %(message)s')
    if cfg.dataset.training_dataset_path == '..\\dataset\\full\\train':
        file_handler = logging.FileHandler('../logs/{}-FULL-{}-{}-{}.log'.format(cfg.network.architecture,
                                                                                 TIMESTAMP,
                                                                                 cfg.network.loss_function,
                                                                                 cfg.network.optimizer))
        if cfg.dataset_config.generate_extended_test_dataset:
            file_handler = logging.FileHandler(
                '../logs/{}-FULL-EXTENDED-{}-{}-{}.log'.format(cfg.network.architecture, TIMESTAMP,
                                                               cfg.network.loss_function,
                                                               cfg.network.optimizer))
    else:
        file_handler = logging.FileHandler('../logs/{}-{}-{}-{}.log'.format(cfg.network.architecture,
                                                                            TIMESTAMP,
                                                                            cfg.network.loss_function,
                                                                            cfg.network.optimizer))

    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    log.addHandler(file_handler)
    log.addHandler(stream_handler)
    return log


def print_cfg(cfg):
    log = logging.getLogger('oct-cnn')
    log.info('Starting CNN training.')
    log.info('Architecture: %s', cfg.network.architecture)
    log.info('Image resolution: %s', cfg.dataset.img_size)
    log.info('Training dataset path: %s', cfg.dataset.training_dataset_path)
    log.info('Test dataset path: %s', cfg.dataset.test_dataset_path)
    if cfg.dataset.generate_extended_test_dataset:
        log.warning('USING EXTENDED TEST DATA GENERATOR')
    log.info('Loss function: %s', cfg.network.loss_function)
    log.info('Optimizer: %s', cfg.network.optimizer)
    log.info('Batch size: %s', cfg.training.batch_size)
    log.info('Epochs: %s', cfg.training.epochs)
