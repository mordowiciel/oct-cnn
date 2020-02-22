from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def CustomNetOne(input_shape, classes):
    model = Sequential()

    # Block 1 (3x3 convolution)
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # Block 2 (factorization 11x11)
    model.add(Conv2D(filters=64, kernel_size=(7, 1)))
    model.add(Conv2D(filters=64, kernel_size=(1, 7), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # Block 3 (factorization 7x7)
    model.add(Conv2D(filters=64, kernel_size=(7, 1)))
    model.add(Conv2D(filters=64, kernel_size=(1, 7), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    # Block 4 (factorization 7x7)
    model.add(Conv2D(filters=64, kernel_size=(7, 1)))
    model.add(Conv2D(filters=64, kernel_size=(1, 7), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # Block 5 (factorization 5x5)
    model.add(Conv2D(filters=128, kernel_size=(5, 1)))
    model.add(Conv2D(filters=128, kernel_size=(1, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    # Block 6 (factorization 5x5)
    model.add(Conv2D(filters=128, kernel_size=(5, 1)))
    model.add(Conv2D(filters=128, kernel_size=(1, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # Block 7 (3x3 convolution)
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    # Block 7 (3x3 convolution)
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(classes, activation='softmax'))

    model.summary()

    return model
