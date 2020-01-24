from keras_preprocessing.image import ImageDataGenerator

from augmentation.augmentation_preprocessor import AugmentationPreprocessor


class GeneratorResolver:

    def __init__(self, cfg, generator_seed=42):
        self.cfg = cfg
        self.generator_seed = generator_seed

    def provide_image_data_generators(self):
        augmentation_preprocessor = AugmentationPreprocessor(augmentation_config=self.cfg.augmentation)
        if self.cfg.augmentation.use_data_augmentation:
            training_image_datagen = ImageDataGenerator(
                horizontal_flip=self.cfg.augmentation.horizontal_flip,
                width_shift_range=self.cfg.augmentation.width_shift_range,
                height_shift_range=self.cfg.augmentation.height_shift_range,
                brightness_range=self.cfg.augmentation.brightness_range,
                rescale=1. / 255,
                preprocessing_function=augmentation_preprocessor.preprocessing_chain
            )
            test_image_datagen = ImageDataGenerator(rescale=1. / 255)
        else:
            training_image_datagen = ImageDataGenerator(rescale=1. / 255)
            test_image_datagen = ImageDataGenerator(rescale=1. / 255)

        return training_image_datagen, test_image_datagen

    def resolve_generators(self):
        training_image_datagen, test_image_datagen = self.provide_image_data_generators()
        if self.cfg.network.architecture == 'VGG-16-TL':
            training_generator = training_image_datagen.flow_from_directory(
                directory=self.cfg.dataset.training_dataset_path,
                target_size=self.cfg.dataset.img_size,
                batch_size=self.cfg.training.training_batch_size,
                interpolation='bilinear',
                color_mode='rgb',
                seed=self.generator_seed,
                shuffle=True
            )
            test_generator = test_image_datagen.flow_from_directory(
                directory=self.cfg.dataset.test_dataset_path,
                target_size=self.cfg.dataset.img_size,
                batch_size=self.cfg.training.test_batch_size,
                interpolation='bilinear',
                color_mode='rgb',
                seed=self.generator_seed,
                shuffle=False
            )
        else:
            training_generator = training_image_datagen.flow_from_directory(
                directory=self.cfg.dataset.training_dataset_path,
                target_size=self.cfg.dataset.img_size,
                batch_size=self.cfg.training.training_batch_size,
                interpolation='bilinear',
                color_mode='grayscale',
                seed=self.generator_seed,
                shuffle=True
            )
            test_generator = test_image_datagen.flow_from_directory(
                directory=self.cfg.dataset.test_dataset_path,
                target_size=self.cfg.dataset.img_size,
                batch_size=self.cfg.training.test_batch_size,
                interpolation='bilinear',
                color_mode='grayscale',
                seed=self.generator_seed,
                shuffle=False
            )

        return training_generator, test_generator
