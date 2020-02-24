import logging
from glob import glob
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from oct_utils.plot_utils import save_confusion_matrix

log = logging.getLogger('oct-cnn')


class ModelEvaluator:
    def __init__(self, cfg, model, test_data_generator):
        self.cfg = cfg
        self.model = model
        self.test_data_generator = test_data_generator

    def __count_images(self, dir_path):
        return len(glob('{}//**//*.jpeg'.format(dir_path), recursive=True))

    def __calculate_steps_per_epoch(self):
        test_image_count = self.__count_images(self.cfg.dataset.test_dataset_path)
        test_steps_per_epoch = test_image_count // self.cfg.training.test_batch_size
        return test_steps_per_epoch

    def __get_item_paths(self):
        return sorted(glob('{}/**/*.jpeg'.format(self.cfg.dataset.test_dataset_path), recursive=True))

    def __result_resolver(self, res):
        concat_res = []
        for i in res:
            highest_class_idx = np.argmax(i)
            concat_res.append(highest_class_idx)
        return concat_res

    def __resolve_item_label(self, filepath):
        class_map = {"CNV": 0, "DME": 1, "DRUSEN": 2, "NORMAL": 3}
        img_path = Path(filepath)
        label = class_map[img_path.name.split('-')[0]]
        return label

    def evaluate_model(self):
        steps_per_epoch = self.__calculate_steps_per_epoch()

        log.info('Evaluate generator...')
        score = self.model.evaluate_generator(self.test_data_generator, steps=steps_per_epoch,
                                              use_multiprocessing=False,
                                              verbose=0)
        log.info('Generator evaluation:')
        log.info('Test loss: %s', score[0])
        log.info('Test accuracy: %s', score[1])
        log.info('Evaluate generator complete.')

        log.info('Starting detailed model evaluation...')
        item_paths = self.__get_item_paths()
        y_true = [self.__resolve_item_label(filepath) for filepath in item_paths]
        y_pred = self.__result_resolver(
            self.model.predict_generator(self.test_data_generator, steps=steps_per_epoch, verbose=0))
        log.info('Detailed model evaluation complete.')

        # Generate classification report
        labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        report = classification_report(y_true, y_pred, target_names=labels)
        log.info('\n' + report)

        # Create and plot confusion matrix
        save_confusion_matrix(y_true, y_pred, self.cfg.misc.logs_path, normalize=False)