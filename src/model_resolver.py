import logging

import keras

from architectures.alexnet import AlexNet
from architectures.customnet import CustomNetOne
from architectures.lenet import LeNet5
from architectures.vgg16_transfer import VGG16TL

log = logging.getLogger('oct-cnn')


class ModelResolver:

    def __init__(self, cfg):
        self.cfg = cfg

    def resolve_model(self):
        if self.cfg.network.architecture == "LeNet":
            raise AttributeError('LeNet not supported, use LeNet5 instead')
        if self.cfg.network.architecture == "LeNet5":
            model = LeNet5(self.cfg.dataset.input_shape, classes=4)
        elif self.cfg.network.architecture == "VGG-16":
            model = keras.applications.vgg16.VGG16(include_top=True, weights=None,
                                                   input_tensor=None, input_shape=self.cfg.dataset.input_shape,
                                                   pooling=None, classes=4)
        elif self.cfg.network.architecture == "VGG-16-TL":
            model = VGG16TL(input_shape=self.cfg.dataset.input_shape, classes=4)
        elif self.cfg.network.architecture == "AlexNet":
            model = AlexNet(input_shape=self.cfg.dataset.input_shape, classes=4)
        elif self.cfg.network.architecture == "CustomNet":
            model = CustomNetOne(input_shape=self.cfg.dataset.input_shape, classes=4)
        else:
            raise AttributeError('Unknown network architecture provided, aborting.')

        model.compile(loss=self.cfg.network.loss_function,
                      optimizer=self.cfg.network.optimizer,
                      metrics=['accuracy', 'mse'])

        log.info('Model summary:')
        model.summary()
        return model
