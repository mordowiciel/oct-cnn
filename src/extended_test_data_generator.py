import glob
import random
from pathlib import Path

import cv2
import keras
import numpy as np
import skimage
from skimage.transform import resize


class ExtendedTestDataGenerator(keras.utils.Sequence):

    def __init__(self, dataset_path, batch_size=32, dim=(496, 512), n_channels=1,
                 n_classes=4, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.item_paths = self.__get_item_paths()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.item_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        item_paths_temp = [self.item_paths[k] for k in indexes]
        x, y = self.__data_generation(item_paths_temp)

        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.item_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, item_paths):
        x = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float16)
        y = np.empty(self.batch_size, dtype=int)
        for counter, filepath in enumerate(item_paths):
            img_arr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) / 255.0
            x[counter] = skimage.transform.resize(img_arr, self.dim + (1,))
            y[counter] = self.__resolve_item_label(filepath)

        return x, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __get_item_paths(self):
        cnv_paths = []
        for counter, filepath in enumerate(
                glob.iglob('{}\\CNV\\*.jpeg'.format(self.dataset_path),
                           recursive=True)):
            cnv_paths.append(filepath)

        dme_paths = []
        for counter, filepath in enumerate(
                glob.iglob('{}\\DME\\*.jpeg'.format(self.dataset_path),
                           recursive=True)):
            dme_paths.append(filepath)

        drusen_paths = []
        for counter, filepath in enumerate(
                glob.iglob('{}\\DRUSEN\\*.jpeg'.format(self.dataset_path),
                           recursive=True)):
            drusen_paths.append(filepath)

        normal_paths = []
        for counter, filepath in enumerate(
                glob.iglob('{}\\NORMAL\\*.jpeg'.format(self.dataset_path),
                           recursive=True)):
            normal_paths.append(filepath)

        # Always generate same random sample
        random.seed(666)
        item_paths = [random.sample(cnv_paths, 2000),
                      random.sample(dme_paths, 2000),
                      random.sample(drusen_paths, 2000),
                      random.sample(normal_paths, 2000)]

        # flatten the list
        return [item for sublist in item_paths for item in sublist]

    def __resolve_item_label(self, filepath):
        class_map = {"CNV": 0, "DME": 1, "DRUSEN": 2, "NORMAL": 3}
        img_path = Path(filepath)
        label = class_map[img_path.name.split('-')[0]]
        return label
