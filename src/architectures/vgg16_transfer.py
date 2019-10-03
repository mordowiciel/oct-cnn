from keras import applications
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model


def VGG16TL(input_shape, classes):
    vgg_model = applications.VGG16(weights='imagenet',
                                   include_top=False,
                                   input_shape=input_shape,
                                   classes=classes)

    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

    # Getting output tensor of the last VGG layer that we want to include
    x = layer_dict['block4_pool'].output

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv1')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Stack last layer together with fully connected classificator layer
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Creating new model. Please note that this is NOT a Sequential() model.
    vgg_16_tl = Model(input=vgg_model.input, output=x)

    # Make sure that the pre-trained bottom layers are not trainable
    for layer in vgg_16_tl.layers[:15]:
        layer.trainable = False

    return vgg_16_tl
