import keras

from architectures.alexnet import AlexNet
from architectures.lenet import LeNet5


def resolve_model(cfg):
    if cfg.network_config.architecture == "LeNet":
        # model = LeNet(cfg.dataset_config.input_shape, KERNEL_SIZE)
        raise AttributeError('LeNet not supported, use LeNet5 instead')
    if cfg.network_config.architecture == "LeNet5":
        model = LeNet5(cfg.dataset_config.input_shape, class_count=4)
    elif cfg.network_config.architecture == "VGG-16":
        model = keras.applications.vgg16.VGG16(include_top=True, weights=None,
                                               input_tensor=None, input_shape=cfg.dataset_config.input_shape,
                                               pooling=None, classes=4)
    elif cfg.network_config.architecture == "AlexNet":
        model = AlexNet(input_shape=cfg.dataset_config.input_shape, classes=4)
    else:
        raise AttributeError('Unknown network architecture provided, aborting.')

    model.compile(loss=cfg.network_config.loss_function,
                  optimizer=cfg.network_config.optimizer,
                  metrics=['categorical_accuracy'])

    return model
