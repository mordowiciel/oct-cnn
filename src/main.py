import datetime
import logging
import sys

from emergency_model_evaluator import manually_evaluate_model
from extended_test_data_generator import ExtendedTestDataGenerator
from model_evaluator import evaluate_model
from model_resolver import resolve_model
from oct_config import OCTConfig
from oct_data_generator import *

''' Notes:
* standard LeNet, full size - dziala na batch_size 20
* VGG16, 2x scaledown - dziala na batch_size 4
'''

cfg = OCTConfig('config.ini')

# Setup logger
log = logging.getLogger('oct-cnn')
log.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s - %(message)s')
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
if cfg.dataset_config.training_dataset_path == '..\\dataset\\full\\train':
    file_handler = logging.FileHandler('../logs/{}-FULL-{}-{}-{}.log'.format(cfg.network_config.architecture,
                                                                             TIMESTAMP,
                                                                             cfg.network_config.loss_function,
                                                                             cfg.network_config.optimizer))
    if cfg.dataset_config.generate_extended_test_dataset:
        file_handler = logging.FileHandler(
            '../logs/{}-FULL-EXTENDED-{}-{}-{}.log'.format(cfg.network_config.architecture, TIMESTAMP,
                                                           cfg.network_config.loss_function,
                                                           cfg.network_config.optimizer))
else:
    file_handler = logging.FileHandler('../logs/{}-{}-{}-{}.log'.format(cfg.network_config.architecture,
                                                                        TIMESTAMP,
                                                                        cfg.network_config.loss_function,
                                                                        cfg.network_config.optimizer))

file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

log.addHandler(file_handler)
log.addHandler(stream_handler)

log.info('Starting CNN training.')
log.info('Architecture: %s', cfg.network_config.architecture)
log.info('Image resolution: %s', cfg.dataset_config.img_size)
log.info('Training dataset path: %s', cfg.dataset_config.training_dataset_path)
log.info('Test dataset path: %s', cfg.dataset_config.test_dataset_path)
if cfg.dataset_config.generate_extended_test_dataset:
    log.warning('USING EXTENDED TEST DATA GENERATOR')
log.info('Loss function: %s', cfg.network_config.loss_function)
log.info('Optimizer: %s', cfg.network_config.optimizer)
log.info('Batch size: %s', cfg.training_config.batch_size)
log.info('Epochs: %s', cfg.training_config.epochs)

if __name__ == '__main__':

    if cfg.dataset_config.generate_extended_test_dataset:
        test_data_generator = ExtendedTestDataGenerator(dataset_path=cfg.dataset_config.training_dataset_path,
                                                        batch_size=cfg.training_config.batch_size,
                                                        dim=cfg.dataset_config.img_size,
                                                        n_channels=1,
                                                        n_classes=4,
                                                        shuffle=True)

        paths_to_skip = test_data_generator.item_paths
        training_data_generator = OCTDataGenerator(dataset_path=cfg.dataset_config.training_dataset_path,
                                                   paths_to_skip=paths_to_skip,
                                                   batch_size=cfg.training_config.batch_size,
                                                   dim=cfg.dataset_config.img_size,
                                                   n_channels=1,
                                                   n_classes=4,
                                                   shuffle=True)

        training_paths = training_data_generator.item_paths
    else:
        test_data_generator = OCTDataGenerator(dataset_path=cfg.dataset_config.test_dataset_path,
                                               batch_size=cfg.training_config.batch_size,
                                               dim=cfg.dataset_config.img_size,
                                               n_channels=1,
                                               n_classes=4,
                                               shuffle=True)
        training_data_generator = OCTDataGenerator(dataset_path=cfg.dataset_config.training_dataset_path,
                                                   batch_size=cfg.training_config.batch_size,
                                                   dim=cfg.dataset_config.img_size,
                                                   n_channels=1,
                                                   n_classes=4,
                                                   shuffle=True)

    model = resolve_model(cfg)
    model.fit_generator(generator=training_data_generator,
                        validation_data=test_data_generator,
                        use_multiprocessing=False,
                        epochs=cfg.training_config.epochs)
    score = model.evaluate_generator(test_data_generator, use_multiprocessing=False,
                                     verbose=0)

    model_path = '..\\models\\{}-{}-{}-{}.h5'.format(cfg.network_config.architecture,
                                                     TIMESTAMP,
                                                     cfg.network_config.loss_function,
                                                     cfg.network_config.optimizer)
    log.info('Saving model at path %s', model_path)
    model.save(model_path)

    log.info('Model training complete.')
    log.info('Training evaluation:')
    log.info('Test loss: %s', score[0])
    log.info('Test accuracy: %s', score[1])
    log.info('Creating confusion matrix and calculating metrics')
    evaluate_model(model, test_data_generator)
    manually_evaluate_model(model, test_data_generator.item_paths, cfg.dataset_config.img_size)


