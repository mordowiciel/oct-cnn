from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def FactorNet(input_shape, classes):
    model = Sequential()

    # Aggressive, vertical pooling
    model.add(Conv2D(filters=64, kernel_size=(1, 11), strides=(1, 2), activation='relu', input_shape=input_shape, name='conv_vertical_1'))
    model.add(Conv2D(filters=64, kernel_size=(1, 11), strides=(1, 2), activation='relu', padding='valid', name='conv_vertical_2'))
    model.add(Conv2D(filters=64, kernel_size=(1, 11), strides=(1, 2), activation='relu', padding='valid', name='conv_vertical_3'))

    # Slower, horizontal pooling
    model.add(Conv2D(filters=128, kernel_size=(17, 1), activation='relu', padding='valid', name='conv_horizontal_1'))
    model.add(Conv2D(filters=128, kernel_size=(17, 1), activation='relu', padding='valid', name='conv_horizontal_2'))
    model.add(Conv2D(filters=128, kernel_size=(17, 1), activation='relu', padding='valid', name='conv_horizontal_3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='maxpool_horizontal_3'))

    model.add(Conv2D(filters=192, kernel_size=(11, 1), activation='relu', padding='valid', name='conv_horizontal_4'))
    model.add(Conv2D(filters=192, kernel_size=(11, 1), activation='relu', padding='valid', name='conv_horizontal_5'))
    model.add(Conv2D(filters=192, kernel_size=(11, 1), activation='relu', padding='valid', name='conv_horizontal_6'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='maxpool_horizontal_6'))

    model.add(Conv2D(filters=192, kernel_size=(7, 1), activation='relu', padding='valid', name='conv_horizontal_7'))
    model.add(Conv2D(filters=192, kernel_size=(7, 1), activation='relu', padding='valid', name='conv_horizontal_8'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='maxpool_horizontal_8'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(classes, activation='softmax'))

    return model