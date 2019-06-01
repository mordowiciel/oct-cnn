import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix

log = logging.getLogger('oct-cnn')


def result_resolver(res):
    concat_res = []
    for i in res:
        highest_class_idx = np.argmax(i)
        concat_res.append(highest_class_idx)
    return concat_res


def construct_confusion_matrix_plot(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, range(4), range(4))
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})  # font size


def resolve_logger_filename():
    log_file_handler = log.handlers[0]
    log_file_path = log_file_handler.baseFile
    return os.path.basename(log_file_path)


def evaluate_model(model, test_data_generator):
    y_true = test_data_generator.item_labels
    y_pred = result_resolver(model.predict_generator(test_data_generator, verbose=1))

    # Generate classification report
    labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    report = classification_report(y_true, y_pred, target_names=labels)
    log.info(report)

    # Create and plot confusion matrix
    construct_confusion_matrix_plot(y_true, y_pred)
    plt.savefig('./logs/cm/%s' % resolve_logger_filename())
