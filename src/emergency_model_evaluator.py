import logging
import os
from pathlib import Path

import cv2
import numpy as np
import skimage

log = logging.getLogger('oct-cnn')


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


def manually_evaluate_model(model, test_data_filepaths, img_size):
    global_count = 0
    global_accurate_count = 0
    class_count_map = {'CNV': 0, 'DME': 0, 'DRUSEN': 0, 'NORMAL': 0}
    accurate_class_count_map = {'CNV': 0, 'DME': 0, 'DRUSEN': 0, 'NORMAL': 0}

    for filepath in test_data_filepaths:
        global_count += 1
        img = np.empty((1, *img_size, 1), dtype=np.float64)
        img_arr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) / 255.0
        img[0] = skimage.transform.resize(img_arr, img_size + (1,))

        res = model.predict(img, verbose=1)
        correct_label = resolve_item_label(filepath)
        class_count_map[correct_label] += 1

        predicted_label = result_resolver(res)

        if correct_label == predicted_label:
            accurate_class_count_map[predicted_label] += 1

        global_accurate_count += 1

        log.debug('Predicition for %s : %s (%s)' % (os.path.basename(filepath), res, predicted_label))
        log.debug('Class count: %s' % class_count_map)
        log.debug('Accurate predictions class count : %s' % accurate_class_count_map)

    log.info('Global data:')
    log.info('Class count: %s' % class_count_map)
    log.info('Accurate predictions class count : %s' % accurate_class_count_map)
    log.info('Percentages of class predictions: %s' % get_class_percentage_info(class_count_map, accurate_class_count_map))

    all = sum(class_count_map.values())
    accurate = sum(accurate_class_count_map.values())
    log.info('Global percentage precision: %s / %s (%.6f %%)' % (accurate, all, accurate / all))
