import datetime

from emergency_model_evaluator import manually_evaluate_model
from model_evaluator import evaluate_model
from model_resolver import resolve_model
from oct_config import OCTConfig
from oct_generator_provider import provide_generators
from oct_logger import setup_logger, print_cfg

''' Notes:
* standard LeNet, full size - dziala na batch_size 20
* VGG16, 2x scaledown - dziala na batch_size 4
'''
if __name__ == '__main__':
    TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    cfg = OCTConfig('config.ini')
    log = setup_logger(cfg, TIMESTAMP)
    print_cfg(cfg)

    training_data_generator, test_data_generator = provide_generators(cfg)

    model = resolve_model(cfg)
    model.fit_generator(generator=training_data_generator,
                        validation_data=test_data_generator,
                        use_multiprocessing=False,
                        epochs=cfg.training.epochs)
    model_path = '..\\models\\{}-{}-{}-{}.h5'.format(cfg.network.architecture,
                                                     TIMESTAMP,
                                                     cfg.network.loss_function,
                                                     cfg.network.optimizer)
    log.info('Model training complete.')
    log.info('Saving model at path %s', model_path)
    model.save(model_path)

    score = model.evaluate_generator(test_data_generator, use_multiprocessing=False,
                                     verbose=0)

    log.info('Training evaluation:')
    log.info('Test loss: %s', score[0])
    log.info('Test accuracy: %s', score[1])
    log.info('Creating confusion matrix and calculating metrics')
    evaluate_model(model, test_data_generator)
    manually_evaluate_model(model, test_data_generator.item_paths, cfg.dataset.img_size)
