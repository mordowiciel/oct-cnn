from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def FactorNet(input_shape, classes):
    model = Sequential()

    # Aggressive, vertical pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding='valid', name='maxpool_vertical_1', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding='valid', name='maxpool_vertical_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding='valid', name='maxpool_vertical_3'))

    # Slower, horizontal pooling
    model.add(Conv2D(filters=128, kernel_size=(17, 1), activation='relu', padding='valid', name='conv_horizontal_1'))
    model.add(Conv2D(filters=128, kernel_size=(17, 1), activation='relu', padding='valid', name='conv_horizontal_2'))
    model.add(Conv2D(filters=128, kernel_size=(17, 1), activation='relu', padding='valid', name='conv_horizontal_3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='maxpool_horizontal_3'))

    model.add(Conv2D(filters=128, kernel_size=(11, 1), activation='relu', padding='valid', name='conv_horizontal_4'))
    model.add(Conv2D(filters=128, kernel_size=(11, 1), activation='relu', padding='valid', name='conv_horizontal_5'))
    model.add(Conv2D(filters=128, kernel_size=(11, 1), activation='relu', padding='valid', name='conv_horizontal_6'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='maxpool_horizontal_6'))

    model.add(Conv2D(filters=128, kernel_size=(7, 1), activation='relu', padding='valid', name='conv_horizontal_7'))
    model.add(Conv2D(filters=128, kernel_size=(7, 1), activation='relu', padding='valid', name='conv_horizontal_8'))
    model.add(Conv2D(filters=128, kernel_size=(7, 1), activation='relu', padding='valid', name='conv_horizontal_9'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='maxpool_horizontal_9'))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(classes, activation='softmax'))

    model.summary()

    return model
