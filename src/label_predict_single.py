'''
CANDIDATE TO DELETE
'''

import glob
import os
from pathlib import Path

import cv2
import numpy as np
import skimage
from keras.models import load_model

IMG_SIZE = (124, 128)
INPUT_SHAPE = IMG_SIZE + (1,)


def get_class_percentage_info(class_count, accurate_class_count):
    prec_info = {}
    for key in class_count:
        global_count_val = class_count[key]
        accurate_count_val = accurate_class_count[key]
        prec_info[key] = accurate_count_val / global_count_val
    return prec_info


def result_resolver(res):
    class_map = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}
    highest_class_idx = np.argmax(res)
    return class_map[highest_class_idx]


def resolve_item_label(filepath):
    img_path = Path(filepath)
    return img_path.name.split('-')[0]


model = load_model('../models/VGG-16-FULL-2019-04-24T23-53-11-categorical_crossentropy-sgd.h5')
global_count = 0
global_accurate_count = 0

class_count_map = {'CNV': 0, 'DME': 0, 'DRUSEN': 0, 'NORMAL': 0}
accurate_class_count_map = {'CNV': 0, 'DME': 0, 'DRUSEN': 0, 'NORMAL': 0}

for filepath in glob.iglob('{}\\**\\*.jpeg'.format('C:/Users/marcinis/Politechnika/sem8/inz/dataset/full/test'),
                           recursive=True):
    global_count += 1

    # TODO: dlaczego jest tak drastyczna roznica w precyzji?
    # TODO: typ pozostaje ten sam
    img = np.empty((1, *IMG_SIZE, 1), dtype=np.float64)

    img_arr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) / 255.0
    img[0] = skimage.transform.resize(img_arr, IMG_SIZE + (1,))

    # # float64
    # img_arr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) / 255.0
    # img = skimage.transform.resize(img_arr, (1,) + IMG_SIZE + (1,))

    res = model.predict(img, verbose=1)
    correct_label = resolve_item_label(filepath)
    class_count_map[correct_label] += 1

    predicted_label = result_resolver(res)

    if correct_label == predicted_label:
        accurate_class_count_map[predicted_label] += 1

    global_accurate_count += 1

    print('Predicition for %s : %s (%s)' % (os.path.basename(filepath), res, predicted_label))
    print('Class count: %s' % class_count_map)
    print('Accurate predictions class count : %s' % accurate_class_count_map)

print()
print('Global data:')
print('Class count: %s' % class_count_map)
print('Accurate predictions class count : %s' % accurate_class_count_map)
print('Percentages of class predictions: %s' % get_class_percentage_info(class_count_map, accurate_class_count_map))

all = sum(class_count_map.values())
accurate = sum(accurate_class_count_map.values())
print('Global percentage precision: %s / %s (%.6f %%)' % (accurate, all, accurate / all))
