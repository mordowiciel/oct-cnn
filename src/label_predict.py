from keras.models import load_model

from oct_data_generator import OCTDataGenerator

IMG_SIZE = (124, 128)
INPUT_SHAPE = IMG_SIZE + (1,)
model = load_model('../models/LENET-FULL-2019-04-27T12-20-57-categorical_crossentropy-sgd.h5')
val_data_generator = OCTDataGenerator(dataset_path='C:/Users/marcinis/Politechnika/sem8/inz/dataset/full/test',
                                      batch_size=1,
                                      dim=IMG_SIZE,
                                      n_channels=1,
                                      n_classes=4,
                                      shuffle=False)

# paths = val_data_generator.item_paths
# data = val_data_generator._data_generation(paths)


# print(val_data_generator.item_paths)
# print(val_data_generator.item_labels)
# item = val_data_generator.__getitem__(0)
# item2 = val_data_generator.__getitem__(31)


res = model.predict_generator(val_data_generator, verbose=1)
# print(res)
#