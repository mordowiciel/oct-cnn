import datetime
import os
from glob import glob

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from augmentation.preprocessing_functions import all_in_once
from batch_history_callback import BatchHistory
from model_evaluator import evaluate_model
from model_resolver import resolve_model
from oct_config import OCTConfig
from oct_logger import setup_logger, print_cfg
from oct_utils.plot_utils import save_loss_to_batch_graph


def count_images(dir_path):
    return len(glob('{}//**//*.jpeg'.format(dir_path), recursive=True))


if __name__ == '__main__':
    TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    cfg = OCTConfig('../config.ini')
    log = setup_logger(cfg, TIMESTAMP)
    print_cfg(cfg)

    log.info('Providing generators...')

    if cfg.augmentation.use_data_augmentation:
        training_datagen = ImageDataGenerator(
            horizontal_flip=cfg.augmentation.horizontal_flip,
            width_shift_range=cfg.augmentation.width_shift_range,
            height_shift_range=cfg.augmentation.height_shift_range,
            brightness_range=cfg.augmentation.brightness_range,
            preprocessing_function=all_in_once
        )
        test_datagen = ImageDataGenerator()
    else:
        training_datagen = ImageDataGenerator()
        test_datagen = ImageDataGenerator()


    #### DEBUG #####

    if not os.path.exists('../preview'):
        os.mkdir('../preview')

    img = load_img('C:/Users/marcinis/Politechnika/sem8/inz/dataset/full/train/CNV/CNV-13823-1.jpeg')
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in training_datagen.flow(x, batch_size=1,
                                       save_to_dir='../preview', save_prefix='cnv', save_format='jpeg'):
        i += 1
        if i > 50:
            break  # otherwise the generator would loop indefinitely

    #### DEBUG #####

    training_generator = training_datagen.flow_from_directory(
        directory=cfg.dataset.training_dataset_path,
        target_size=cfg.dataset.img_size,
        batch_size=cfg.training.training_batch_size,
        color_mode='grayscale',
        shuffle=True
    )
    test_generator = test_datagen.flow_from_directory(
        directory=cfg.dataset.test_dataset_path,
        target_size=cfg.dataset.img_size,
        batch_size=cfg.training.test_batch_size,
        color_mode='grayscale',
        shuffle=False
    )

    log.info('Resolving model...')
    model = resolve_model(cfg)

    log.info('Fitting model...')
    batch_history = BatchHistory(granularity=100)
    training_steps_per_epoch = count_images(cfg.dataset.training_dataset_path) // cfg.training.training_batch_size
    history = model.fit_generator(generator=training_generator,
                                  use_multiprocessing=False,
                                  epochs=cfg.training.epochs,
                                  callbacks=[batch_history],
                                  steps_per_epoch=training_steps_per_epoch)

    log.info('Model training complete.')
    log.info('Saving loss to batch graph.')
    save_loss_to_batch_graph(batch_history.history['batch'],
                             batch_history.history['loss'],
                             cfg.misc.logs_path)

    model_path = '{}//{}-{}-{}-{}.h5'.format(cfg.misc.models_path,
                                             cfg.network.architecture,
                                             TIMESTAMP,
                                             cfg.network.loss_function,
                                             cfg.network.optimizer)
    log.info('Saving model at path %s', model_path)
    model.save(model_path)

    image_count = count_images(cfg.dataset.test_dataset_path)
    test_steps_per_epoch = image_count // cfg.training.test_batch_size
    score = model.evaluate_generator(test_generator, steps=test_steps_per_epoch, use_multiprocessing=False,
                                     verbose=0)

    log.info('Training evaluation:')
    log.info('Test loss: %s', score[0])
    log.info('Test accuracy: %s', score[1])
    
    # model = load_model(
    #     'C:/Users/marcinis/Politechnika/sem8/inz/final_models/VGG-16-2019-10-08T18-08-01-categorical_crossentropy-sgd.h5'
    # )
    evaluate_model(model, test_generator, cfg.dataset.test_dataset_path, test_steps_per_epoch, cfg.misc.logs_path)
    # manually_evaluate_model(cfg, model, cfg.dataset.test_dataset_path, cfg.dataset.img_size)
