import keras

from architectures.alexnet import AlexNet
from architectures.lenet import LeNet5


def resolve_model(cfg):
    if cfg.network.architecture == "LeNet":
        # model = LeNet(cfg.dataset.input_shape, KERNEL_SIZE)
        raise AttributeError('LeNet not supported, use LeNet5 instead')
    if cfg.network.architecture == "LeNet5":
        model = LeNet5(cfg.dataset.input_shape, class_count=4)
    elif cfg.network.architecture == "VGG-16":
        model = keras.applications.vgg16.VGG16(include_top=True, weights=None,
                                               input_tensor=None, input_shape=cfg.dataset.input_shape,
                                               pooling=None, classes=4)
    elif cfg.network.architecture == "AlexNet":
        model = AlexNet(input_shape=cfg.dataset.input_shape, classes=4)
    else:
        raise AttributeError('Unknown network architecture provided, aborting.')

    model.compile(loss=cfg.network.loss_function,
                  optimizer=cfg.network.optimizer,
                  metrics=['categorical_accuracy'])

    return model
