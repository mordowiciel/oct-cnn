import datetime
from glob import glob

import wandb
from keras.preprocessing.image import ImageDataGenerator

from augmentation.augmentation_preprocessor import AugmentationPreprocessor
from callbacks.batch_history_callback import BatchHistory
from callbacks.time_history_callback import TimeHistory
from generator_resolver import resolve_generators
from model_evaluator import evaluate_model
from model_resolver import resolve_model
from oct_config import OCTConfig
from oct_logger import setup_logger, print_cfg
from oct_utils.plot_utils import save_loss_to_batch_graph


def count_images(dir_path):
    return len(glob('{}//**//*.jpeg'.format(dir_path), recursive=True))


RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

if __name__ == '__main__':

    # Setup config
    cfg = OCTConfig('../config.ini')
    print_cfg(cfg)

    # Setup metrics collection by wandb
    wandb_run_name = '{}-{}-{}-{}'.format(cfg.network.architecture, RUN_TIMESTAMP, cfg.network.loss_function,
                                          cfg.network.optimizer)
    wandb.init(name=wandb_run_name)

    # Setup logger
    log = setup_logger(cfg, RUN_TIMESTAMP)

    augmentation_preprocessor = AugmentationPreprocessor(augmentation_config=cfg.augmentation,
                                                         preprocessing_functions=['gaussian_noise', 'contrast'])
    if cfg.augmentation.use_data_augmentation:
        training_image_datagen = ImageDataGenerator(
            horizontal_flip=cfg.augmentation.horizontal_flip,
            width_shift_range=cfg.augmentation.width_shift_range,
            height_shift_range=cfg.augmentation.height_shift_range,
            brightness_range=cfg.augmentation.brightness_range,
            rescale=1. / 255,
            preprocessing_function=augmentation_preprocessor.preprocessing_chain
        )
        test_image_datagen = ImageDataGenerator(rescale=1. / 255)
    else:
        training_image_datagen = ImageDataGenerator(rescale=1. / 255)
        test_image_datagen = ImageDataGenerator(rescale=1. / 255)

    #### DEBUG #####

    # if not os.path.exists('../preview'):
    #     os.mkdir('../preview')
    #
    # img = load_img('C:/Users/marcinis/Politechnika/sem8/inz/dataset/full/train/CNV/CNV-13823-1.jpeg',
    #                color_mode='grayscale')
    # x = img_to_array(img)
    # x = x.reshape((1,) + x.shape)
    # i = 0
    # for batch in training_datagen.flow(x, batch_size=1,
    #                                    save_to_dir='../preview', save_prefix='cnv', save_format='jpeg'):
    #     i += 1
    #     if i > 50:
    #         break  # otherwise the generator would loop indefinitely

    #### DEBUG #####

    training_generator, test_generator = resolve_generators(cfg, training_image_datagen, test_image_datagen)

    log.info('Resolving model...')
    model = resolve_model(cfg)
    log.info('Model summary:')
    model.summary()

    log.info('Fitting model...')
    batch_history = BatchHistory(granularity=100)
    time_history = TimeHistory()
    training_steps_per_epoch = count_images(cfg.dataset.training_dataset_path) // cfg.training.training_batch_size
    log.info('Training steps per epoch: %s', training_steps_per_epoch)

    history = model.fit_generator(generator=training_generator,
                                  use_multiprocessing=False,
                                  epochs=cfg.training.epochs,
                                  callbacks=[batch_history, time_history],
                                  steps_per_epoch=training_steps_per_epoch)

    log.info('Model training complete.')
    log.info('Saving loss to batch graph.')
    log.info('EPOCH TRAINING TIMES: %s', time_history.epochs_training_duration)
    save_loss_to_batch_graph(batch_history.history['batch'],
                             batch_history.history['loss'],
                             cfg.misc.logs_path)

    model_path = '{}//{}-{}-{}-{}.h5'.format(cfg.misc.models_path,
                                             cfg.network.architecture,
                                             RUN_TIMESTAMP,
                                             cfg.network.loss_function,
                                             cfg.network.optimizer)
    log.info('Saving model at path %s', model_path)
    model.save(model_path)

    image_count = count_images(cfg.dataset.test_dataset_path)
    log.info('Image count %s', image_count)
    test_steps_per_epoch = image_count // cfg.training.test_batch_size
    log.info('Test steps per epoch %s', test_steps_per_epoch)

    score = model.evaluate_generator(test_generator, steps=test_steps_per_epoch, use_multiprocessing=False,
                                     verbose=0)

    log.info('Training evaluation:')
    log.info('Test loss: %s', score[0])
    log.info('Test accuracy: %s', score[1])

    evaluate_model(model, test_generator, cfg.dataset.test_dataset_path, test_steps_per_epoch, cfg.misc.logs_path)
    # manually_evaluate_model(cfg, model, cfg.dataset.test_dataset_path, cfg.dataset.img_size)
