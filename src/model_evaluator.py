import logging
import os

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


# def construct_confusion_matrix_plot(y_true, y_pred):
#     cm = confusion_matrix(y_true, y_pred)
#     df_cm = pd.DataFrame(cm, range(4), range(4))
#     plt.figure(figsize=(10, 7))
#     sn.set(font_scale=1.4)  # for label size
#     sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})  # font size


# def plot_confusion_matrix(y_true, y_pred, normalize):
#     cm = confusion_matrix(y_true, y_pred)
#     classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#     fig, ax = plt.subplots()
#     im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     ax.figure.colorbar(im, ax=ax)
#
#     # We want to show all ticks...
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            # ... and label them with the respective list entries
#            xticklabels=classes, yticklabels=classes,
#            ylabel='True label',
#            xlabel='Predicted label')
#
#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
#
#     # Loop over data dimensions and create text annotations.
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
#
#     return ax


def resolve_logger_filename():
    log_file_handler = log.handlers[0]
    log_file_path = log_file_handler.baseFilename
    return os.path.splitext(os.path.basename(log_file_path))[0]


def evaluate_model(model, test_data_generator, logs_dir):
    y_true = test_data_generator.item_labels

    log.info('Starting model evaluation...')
    y_pred = result_resolver(model.predict_generator(test_data_generator, verbose=0))
    log.info('Model evaluation complete.')

    # Generate classification report
    labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    report = classification_report(y_true, y_pred, target_names=labels)
    log.info(report)

    # Create and plot confusion matrix
    # construct_confusion_matrix_plot(y_true, y_pred)
    save_confusion_matrix(y_true, y_pred, logs_dir, normalize=True)
    # log.info('Plotting confusion matrix...')
    # plot_confusion_matrix(y_true, y_pred, normalize=True)
    #
    # # Before saving the plot, check if cm directory exists
    # if not os.path.exists('../logs/cm'):
    #     os.mkdir('../logs/cm')
    #
    # # Save the plot
    # plt.savefig('../logs/cm/%s' % resolve_logger_filename())
    #
    # path = '../logs/cm/%s' % resolve_logger_filename()
    # log.info('Saved confusion matrix to %s' % path)
