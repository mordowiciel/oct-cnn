import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix

log = logging.getLogger('oct-cnn')


def __save_plot_to_wandb(name, plt):
    wandb.log({name: wandb.Image(plt)})
    log.info("Synced %s plot with wandb" % name)

def __resolve_logger_filename():
    log_file_handler = log.handlers[0]
    log_file_path = log_file_handler.baseFilename
    return os.path.splitext(os.path.basename(log_file_path))[0]


def save_mse_to_epoch_graph(history, logs_dir):

    mse_y = history.history['mean_squared_error']
    epoch_x = np.arange(1, len(mse_y) + 1)

    plt.plot(epoch_x, mse_y)
    plt.ylabel('MSE')
    plt.xlabel('epoch')

    # Treats X axis as integer
    plt.xticks(epoch_x)

    plot_dir = os.path.join(logs_dir, 'epoch_loss_plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    save_path = os.path.join(plot_dir, __resolve_logger_filename())
    plt.savefig(save_path)
    log.info('Saved loss to epoch graph to %s' % save_path)

    __save_plot_to_wandb("mse_to_epoch", plt)


def save_loss_to_batch_graph(x_batches, y_losses, logs_dir):
    plt.plot(x_batches, y_losses)
    plt.ylabel('loss')
    plt.xlabel('batch')

    plot_dir = os.path.join(logs_dir, 'loss_plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    save_path = os.path.join(plot_dir, __resolve_logger_filename())
    plt.savefig(save_path)
    log.info('Saved loss to batch graph to %s' % save_path)

    __save_plot_to_wandb("loss_to_batch", plt)


def save_confusion_matrix(y_true, y_pred, logs_dir, normalize):
    log.info('Plotting confusion matrix...')
    __construct_confusion_matrix(y_true, y_pred, normalize)

    # Before saving the plot, check if cm directory exists
    plot_dir = os.path.join(logs_dir, 'cm_plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Save the plot
    save_path = os.path.join(plot_dir, __resolve_logger_filename())
    plt.savefig(save_path)
    log.info('Saved confusion matrix to %s' % save_path)

    __save_plot_to_wandb("confusion_matrix", plt)


def __construct_confusion_matrix(y_true, y_pred, normalize):
    cm = confusion_matrix(y_true, y_pred)
    classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return ax
