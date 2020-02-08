from keras_preprocessing.image import ImageDataGenerator


class GeneratorResolver:

    def __init__(self, cfg, generator_seed=42):
        self.cfg = cfg
        self.generator_seed = generator_seed

    def provide_image_data_generators(self):
        training_image_datagen = ImageDataGenerator(rescale=1. / 255)
        test_image_datagen = ImageDataGenerator(rescale=1. / 255)
        val_image_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=self.cfg.dataset.validation_split)
        return training_image_datagen, test_image_datagen, val_image_datagen

    def resolve_data_iterators(self):
        training_image_datagen, test_image_datagen, val_image_datagen = self.provide_image_data_generators()
        if self.cfg.network.architecture == 'VGG-16-TL':
            training_data_iterator = training_image_datagen.flow_from_directory(
                directory=self.cfg.dataset.training_dataset_path,
                target_size=self.cfg.dataset.img_size,
                batch_size=self.cfg.training.training_batch_size,
                interpolation='bilinear',
                color_mode='rgb',
                subset='training',
                seed=self.generator_seed,
                shuffle=True
            )
            test_data_iterator = test_image_datagen.flow_from_directory(
                directory=self.cfg.dataset.test_dataset_path,
                target_size=self.cfg.dataset.img_size,
                batch_size=self.cfg.training.test_batch_size,
                interpolation='bilinear',
                color_mode='rgb',
                seed=self.generator_seed,
                shuffle=False
            )
            val_data_iterator = val_image_datagen.flow_from_directory(
                directory=self.cfg.dataset.training_dataset_path,
                target_size=self.cfg.dataset.img_size,
                batch_size=self.cfg.training.training_batch_size,
                interpolation='bilinear',
                color_mode='rgb',
                subset='validation',
                seed=self.generator_seed,
                shuffle=True
            )
        else:
            training_data_iterator = training_image_datagen.flow_from_directory(
                directory=self.cfg.dataset.training_dataset_path,
                target_size=self.cfg.dataset.img_size,
                batch_size=self.cfg.training.training_batch_size,
                interpolation='bilinear',
                color_mode='grayscale',
                subset='training',
                seed=self.generator_seed,
                shuffle=True
            )
            test_data_iterator = test_image_datagen.flow_from_directory(
                directory=self.cfg.dataset.test_dataset_path,
                target_size=self.cfg.dataset.img_size,
                batch_size=self.cfg.training.test_batch_size,
                interpolation='bilinear',
                color_mode='grayscale',
                seed=self.generator_seed,
                shuffle=False
            )
            val_data_iterator = val_image_datagen.flow_from_directory(
                directory=self.cfg.dataset.training_dataset_path,
                target_size=self.cfg.dataset.img_size,
                batch_size=self.cfg.training.training_batch_size,
                interpolation='bilinear',
                color_mode='grayscale',
                subset='validation',
                seed=self.generator_seed,
                shuffle=True
            )

        return training_data_iterator, test_data_iterator, val_data_iterator
