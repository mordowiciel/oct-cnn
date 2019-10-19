import logging
import os
from glob import glob
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from oct_utils.plot_utils import save_confusion_matrix

log = logging.getLogger('oct-cnn')


def result_resolver(res):
    concat_res = []
    for i in res:
        highest_class_idx = np.argmax(i)
        concat_res.append(highest_class_idx)
    return concat_res


def resolve_logger_filename():
    log_file_handler = log.handlers[0]
    log_file_path = log_file_handler.baseFilename
    return os.path.splitext(os.path.basename(log_file_path))[0]


def __get_item_paths(dataset_path):
    return glob('{}\\**\\*.jpeg'.format(dataset_path), recursive=True)


def __resolve_item_label(filepath):
    class_map = {"CNV": 0, "DME": 1, "DRUSEN": 2, "NORMAL": 3}
    img_path = Path(filepath)
    label = class_map[img_path.name.split('-')[0]]
    return label


def evaluate_model(model, test_data_generator, test_data_dir, steps_per_epoch, logs_dir):
    log.info('Starting model evaluation...')
    item_paths = __get_item_paths(test_data_dir)
    y_true = [__resolve_item_label(filepath) for filepath in item_paths]
    y_pred = result_resolver(model.predict_generator(test_data_generator, steps=steps_per_epoch, verbose=0))
    log.info('Model evaluation complete.')

    # Generate classification report
    labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    report = classification_report(y_true, y_pred, target_names=labels)
    log.info('\n' + report)

    # Create and plot confusion matrix
    save_confusion_matrix(y_true, y_pred, logs_dir, normalize=True)
