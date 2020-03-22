from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def FactorNet(input_shape, classes):
    model = Sequential()

    # Vertical pooling
    model.add(Conv2D(filters=32, kernel_size=(17, 1), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=(1, 17), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding='same'))

    model.add(Conv2D(filters=32, kernel_size=(11, 1), activation='relu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(1, 11), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding='same'))

    model.add(Conv2D(filters=32, kernel_size=(7, 1), activation='relu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(1, 7), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding='same'))

    # Horizontal pooling
    model.add(Conv2D(filters=64, kernel_size=(11, 1), activation='relu', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(1, 11), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='same'))

    model.add(Conv2D(filters=64, kernel_size=(7, 1), activation='relu', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(1, 7), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='same'))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                     name='conv_3x3_1'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                 name='conv_3x3_2'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                 name='conv_3x3_3'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool_3x3_3'))

    model.add(Conv2D(filters=196, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                     name='conv_3x3_4'))
    model.add(Conv2D(filters=196, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                     name='conv_3x3_5'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                     name='conv_3x3_6'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool_3x3_6'))

    model.add(Conv2D(filters=196, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                     name='conv_3x3_7'))
    model.add(Conv2D(filters=196, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                     name='conv_3x3_8'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                     name='conv_3x3_9'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool_3x3_9'))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(classes, activation='softmax'))

    return model
