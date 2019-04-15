from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import logging
import datetime
import sys
from octdatagenerator import *

''' Notes:
* standard LeNet, full size - dziala na batch_size 20
* VGG16, 2x scaledown - dziala na batch_size 4
'''

# Setup logger
log = logging.getLogger('oct-cnn')
log.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s - %(message)s')

datetime = datetime.datetime.now()
TIMESTAMP = datetime.strftime("%Y-%m-%dT%H-%M-%S")
file_handler = logging.FileHandler('../logs/{}.log'.format(TIMESTAMP))
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

log.addHandler(file_handler)
log.addHandler(stream_handler)

training_data_path = '..\\dataset\\oneres\\train'
test_data_path = '..\\dataset\\oneres\\test'

# TODO: move variables to some config?
# IMG_SIZE = (248, 256)
# INPUT_SHAPE = (248, 256, 1)
IMG_SIZE = (124, 128)
INPUT_SHAPE = (124, 128, 1)
BATCH_SIZE = 20
EPOCHS = 5
ARCHITECTURE = "LENET"
LOSS = 'mean_squared_error'
OPTIMIZER = 'rmsprop'

# Use with LENET only
KERNEL_SIZE = (3, 3)

log.info('Starting CNN training.')
log.info('Architecture: %s', ARCHITECTURE)
log.info('Image resolution: %s', IMG_SIZE)
log.info('Loss function: %s', LOSS)
log.info('Optimizer: %s', OPTIMIZER)
log.info('Batch size: %s', BATCH_SIZE)
log.info('Epochs: %s', EPOCHS)

if __name__ == '__main__':
    training_data_generator = OCTDataGenerator(dataset_path=training_data_path,
                                               batch_size=BATCH_SIZE,
                                               dim=IMG_SIZE,
                                               n_channels=1,
                                               n_classes=4,
                                               shuffle=True)

    test_data_generator = OCTDataGenerator(dataset_path=test_data_path,
                                           batch_size=BATCH_SIZE,
                                           dim=IMG_SIZE,
                                           n_channels=1,
                                           n_classes=4,
                                           shuffle=True)

    model = Sequential()
    model.add(Conv2D(20, kernel_size=KERNEL_SIZE, padding="same",
                     input_shape=INPUT_SHAPE,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(50, kernel_size=KERNEL_SIZE, padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    # # TODO: comment above and uncomment below for VGG16
    # model = keras.applications.vgg16.VGG16(include_top=True, weights=None, input_tensor=None,
    #                                        input_shape=INPUT_SHAPE,
    #                                        pooling=None, classes=4)

    model.compile(loss=LOSS,
                  optimizer=OPTIMIZER,
                  metrics=['accuracy'])
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
