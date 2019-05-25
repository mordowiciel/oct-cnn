import datetime
import logging
import sys

from alexnet import AlexNet
from extended_test_data_generator import ExtendedTestDataGenerator
from lenet import LeNet, LeNet5
from oct_data_generator import *

''' Notes:
* standard LeNet, full size - dziala na batch_size 20
* VGG16, 2x scaledown - dziala na batch_size 4
'''

# TODO: move variables to some config?
IMG_SIZE = (124, 128)
INPUT_SHAPE = IMG_SIZE + (1,)
BATCH_SIZE = 20
EPOCHS = 5
ARCHITECTURE = "LeNet"
LOSS = 'categorical_crossentropy'
OPTIMIZER = 'adam'
TRAINING_DATA_PATH = '..\\dataset\\full\\train'
TEST_DATA_PATH = '..\\dataset\\full\\test'
EXTENDED_TEST_DATA = False

# Use with LENET only
KERNEL_SIZE = (3, 3)

# Setup logger
log = logging.getLogger('oct-cnn')
log.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s - %(message)s')
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
if TRAINING_DATA_PATH == '..\\dataset\\full\\train':
    file_handler = logging.FileHandler('../logs/{}-FULL-{}-{}-{}.log'.format(ARCHITECTURE, TIMESTAMP, LOSS, OPTIMIZER))
    if EXTENDED_TEST_DATA:
        file_handler = logging.FileHandler(
            '../logs/{}-FULL-EXTENDED-{}-{}-{}.log'.format(ARCHITECTURE, TIMESTAMP, LOSS, OPTIMIZER))
else:
    file_handler = logging.FileHandler('../logs/{}-{}-{}-{}.log'.format(ARCHITECTURE, TIMESTAMP, LOSS, OPTIMIZER))

file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

log.addHandler(file_handler)
log.addHandler(stream_handler)

log.info('Starting CNN training.')
log.info('Architecture: %s', ARCHITECTURE)
log.info('Image resolution: %s', IMG_SIZE)
log.info('Training dataset path: %s', TRAINING_DATA_PATH)
log.info('Test dataset path: %s', TEST_DATA_PATH)
if EXTENDED_TEST_DATA:
    log.warning('USING EXTENDED TEST DATA GENERATOR')
log.info('Loss function: %s', LOSS)
log.info('Optimizer: %s', OPTIMIZER)
log.info('Batch size: %s', BATCH_SIZE)
log.info('Epochs: %s', EPOCHS)

if __name__ == '__main__':

    if EXTENDED_TEST_DATA:
        test_data_generator = ExtendedTestDataGenerator(dataset_path=TRAINING_DATA_PATH,
                                                        batch_size=BATCH_SIZE,
                                                        dim=IMG_SIZE,
                                                        n_channels=1,
                                                        n_classes=4,
                                                        shuffle=True)

        paths_to_skip = test_data_generator.item_paths
        training_data_generator = OCTDataGenerator(dataset_path=TRAINING_DATA_PATH,
                                                   paths_to_skip=paths_to_skip,
                                                   batch_size=BATCH_SIZE,
                                                   dim=IMG_SIZE,
                                                   n_channels=1,
                                                   n_classes=4,
                                                   shuffle=True)

        training_paths = training_data_generator.item_paths
    else:
        test_data_generator = OCTDataGenerator(dataset_path=TEST_DATA_PATH,
                                               batch_size=BATCH_SIZE,
                                               dim=IMG_SIZE,
                                               n_channels=1,
                                               n_classes=4,
                                               shuffle=True)
        training_data_generator = OCTDataGenerator(dataset_path=TRAINING_DATA_PATH,
                                                   batch_size=BATCH_SIZE,
                                                   dim=IMG_SIZE,
                                                   n_channels=1,
                                                   n_classes=4,
                                                   shuffle=True)

    if ARCHITECTURE == "LeNet":
        model = LeNet(INPUT_SHAPE, KERNEL_SIZE)
    if ARCHITECTURE == "LeNet5":
        model = LeNet5(INPUT_SHAPE, class_count=4)
    elif ARCHITECTURE == "VGG-16":
        model = keras.applications.vgg16.VGG16(include_top=True, weights=None,
                                               input_tensor=None, input_shape=INPUT_SHAPE,
                                               pooling=None, classes=4)
    elif ARCHITECTURE == "AlexNet":
        model = AlexNet(input_shape=INPUT_SHAPE, classes=4)
    else:
        raise AttributeError('Unknown network architecture provided, aborting.')

    log.info(model.summary())
    model.compile(loss=LOSS,
                  optimizer=OPTIMIZER,
                  metrics=['categorical_accuracy'])

    model.fit_generator(generator=training_data_generator,
                        validation_data=test_data_generator,
                        use_multiprocessing=False,
                        epochs=EPOCHS)

    score = model.evaluate_generator(test_data_generator, use_multiprocessing=False,
                                     verbose=0)

    log.info('Model training complete.')
    log.info('Training evaluation:')
    log.info('Test loss: %s', score[0])
    log.info('Test accuracy: %s', score[1])

    model_path = '..\\models\\{}-{}-{}-{}.h5'.format(ARCHITECTURE, TIMESTAMP, LOSS, OPTIMIZER)

    log.info('Saving model at path %s', model_path)
    model.save(model_path)
