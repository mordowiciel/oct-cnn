from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def LeNet(input_shape, kernel_size):
    model = Sequential()
    model.add(Conv2D(20, kernel_size=kernel_size, padding="same",
                     input_shape=input_shape,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(50, kernel_size=kernel_size, padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    return model
