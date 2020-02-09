import logging
import math
import os
import shutil
import sys

from augmentation.augmentation_preprocessor import AugmentationPreprocessor

log = logging.getLogger('oct-cnn')


class AugmentationProcessor:
    def __init__(self, config, training_data_generator):
        self.cfg = config
        self.augmentation_preprocessor = AugmentationPreprocessor(config.augmentation)
        self.training_data_generator = training_data_generator

    def __get_batch_iteration_count(self, image_class):
        training_class_dir = os.path.join(self.cfg.dataset.training_dataset_path, image_class)
        image_with_class_count = len(
            [name for name in os.listdir(training_class_dir) if
             os.path.isfile(os.path.join(training_class_dir, name))])
        final_image_count = int((self.cfg.augmentation.augmentation_count_factor * image_with_class_count))
        return int(math.ceil(final_image_count / self.cfg.augmentation.augmentation_batch_size))

    def __copy_augmented_data_to_training_dataset(self, image_class, augmented_class_dir):
        dest_dir = os.path.join(self.cfg.dataset.training_dataset_path, image_class, 'aug')
        shutil.copytree(augmented_class_dir, dest_dir)

    def perform_data_augmentation(self):

        if not self.cfg.augmentation.use_data_augmentation:
            log.info("Data augmentation is disabled, skipping augmentation step.")
            return

        for image_class in self.cfg.augmentation.classes_to_augment:

            log.info('Performing augmentation for class %s' % image_class)
            augmented_class_dir = os.path.join(self.cfg.augmentation.augmented_images_tmp_save_path, image_class)
            batch_iteration_count = self.__get_batch_iteration_count(image_class)

            if not os.path.exists(augmented_class_dir):
                os.makedirs(augmented_class_dir)
            else:
                log.info(
                    'Directory for given image class already contains augmented data, '
                    'skipping data augmentation for class %s' % image_class)
                return

            dir_iterator = self.training_data_generator.flow_from_directory(
                target_size=self.cfg.dataset.img_size,
                directory=self.cfg.dataset.training_dataset_path,
                batch_size=self.cfg.augmentation.augmentation_batch_size,
                classes=[image_class],
                save_to_dir=augmented_class_dir,
                save_format='jpeg',
                seed=42,
                shuffle=True
            )
            i = 0
            for batch in dir_iterator:
                i += 1
                if i > batch_iteration_count:
                    break
                # sys.stdout.write('\rProcessing batch %d of %d...' % (i, batch_iteration_count))
                # sys.stdout.flush()

            self.__copy_augmented_data_to_training_dataset(image_class, augmented_class_dir)
            log.info('Augmenting data for class %s completed.' % image_class)
