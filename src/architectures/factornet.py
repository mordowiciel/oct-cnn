from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def FactorNet(input_shape, classes):
    model = Sequential()


    ### Vertical factorization N = 7 ###
    model.add(Conv2D(filters=64, kernel_size=(1, 7), activation='relu', input_shape=input_shape, name='conv_vertical_1'))
    model.add(Conv2D(filters=64, kernel_size=(1, 7), activation='relu', padding='valid', name='conv_vertical_2'))
    model.add(Conv2D(filters=64, kernel_size=(1, 7), activation='relu', padding='valid', name='conv_vertical_3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='maxpool_vertical_3'))

    ### Vertical factorization N = 5 ###
    model.add(Conv2D(filters=64, kernel_size=(1, 5), activation='relu', name='conv_vertical_4'))
    model.add(Conv2D(filters=64, kernel_size=(1, 5), activation='relu', padding='valid', name='conv_vertical_5'))
    model.add(Conv2D(filters=64, kernel_size=(1, 5), activation='relu', padding='valid', name='conv_vertical_6'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='maxpool_vertical_6'))

    ### Horizontal factorization N = 7 ###
    model.add(Conv2D(filters=128, kernel_size=(7, 1), activation='relu', padding='valid', name='conv_horizontal_1'))
    model.add(Conv2D(filters=128, kernel_size=(7, 1), activation='relu', padding='valid', name='conv_horizontal_2'))
    model.add(Conv2D(filters=128, kernel_size=(7, 1), activation='relu', padding='valid', name='conv_horizontal_3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='maxpool_horizontal_3'))

    ### Horizontal factorization N = 5 ###
    model.add(Conv2D(filters=128, kernel_size=(5, 1), activation='relu', padding='valid', name='conv_horizontal_4'))
    model.add(Conv2D(filters=128, kernel_size=(5, 1), activation='relu', padding='valid', name='conv_horizontal_5'))
    model.add(Conv2D(filters=128, kernel_size=(5, 1), activation='relu', padding='valid', name='conv_horizontal_6'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='maxpool_horizontal_6'))

    # ### 3x3 blocks ####
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='maxpool_3x3_6'))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(classes, activation='softmax'))

    model.summary()

    return model

# def FactorNet(input_shape, classes):
#     model = Sequential()
#
#     # Block 1 (3x3 convolution)
#     model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
#                      input_shape=input_shape))
#     model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
#
#     # Block 2 (factorization 11x11)
#     model.add(Conv2D(filters=64, kernel_size=(7, 1)))
#     model.add(Conv2D(filters=64, kernel_size=(1, 7), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
#
#     # Block 3 (factorization 7x7)
#     model.add(Conv2D(filters=64, kernel_size=(7, 1)))
#     model.add(Conv2D(filters=64, kernel_size=(1, 7), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
#
#     # Block 4 (factorization 7x7)
#     model.add(Conv2D(filters=64, kernel_size=(7, 1)))
#     model.add(Conv2D(filters=64, kernel_size=(1, 7), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
#
#     # Block 5 (factorization 5x5)
#     model.add(Conv2D(filters=128, kernel_size=(5, 1)))
#     model.add(Conv2D(filters=128, kernel_size=(1, 5), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
#
#     # Block 6 (factorization 5x5)
#     model.add(Conv2D(filters=128, kernel_size=(5, 1)))
#     model.add(Conv2D(filters=128, kernel_size=(1, 5), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
#
#     # Block 7 (3x3 convolution)
#     model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
#                      input_shape=input_shape))
#     model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
#
#     # Block 7 (3x3 convolution)
#     model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
#                      input_shape=input_shape))
#     model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
#
#     # Fully connected layers
#     model.add(Flatten())
#     model.add(Dense(2048, activation='relu'))
#     model.add(Dense(4096, activation='relu'))
#     model.add(Dense(4096, activation='relu'))
#     model.add(Dense(classes, activation='softmax'))
#
#     model.summary()
#
#     return model
