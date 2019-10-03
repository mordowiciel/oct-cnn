import datetime

from batch_history_callback import BatchHistory
from emergency_model_evaluator import manually_evaluate_model
from model_evaluator import evaluate_model
from model_resolver import resolve_model
from oct_config import OCTConfig
from oct_generator_provider import provide_generators
from oct_logger import setup_logger, print_cfg
from oct_utils.plot_utils import save_loss_to_batch_graph

if __name__ == '__main__':
    TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    cfg = OCTConfig('../config.ini')
    log = setup_logger(cfg, TIMESTAMP)
    print_cfg(cfg)

    log.info('Providing generators...')
    training_data_generator, test_data_generator = provide_generators(cfg)

    log.info('Resolving model...')
    model = resolve_model(cfg)

    log.info('Fitting model...')
    batch_history = BatchHistory(granularity=100)
    history = model.fit_generator(generator=training_data_generator,
                                  # validation_data=test_data_generator,
                                  use_multiprocessing=False,
                                  epochs=cfg.training.epochs,
                                  callbacks=[batch_history])
    log.info('Model training complete.')

    # Prepare training accuracy plot
    # Keys that can be retrieved from History object: dict_keys(['loss', 'acc', 'mean_squared_error'])
    # log.info('Keys that can be retrieved from History object %s' % history.history.keys())
    # log.info('Keys that can be retrieved from BatchHistory object %s' % batch_history.history.keys())
    # plt.plot(batch_history.history['batch'], batch_history.history['loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('batch')
    # plt.show()
    save_loss_to_batch_graph(batch_history.history['batch'],
                             batch_history.history['loss'],
                             cfg.misc.logs_path)

    model_path = '{}\\{}-{}-{}-{}.h5'.format(cfg.misc.models_path,
                                             cfg.network.architecture,
                                             TIMESTAMP,
                                             cfg.network.loss_function,
                                             cfg.network.optimizer)
    log.info('Saving model at path %s', model_path)
    model.save(model_path)

    score = model.evaluate_generator(test_data_generator, use_multiprocessing=False,
                                     verbose=0)

    log.info('Training evaluation:')
    log.info('Test loss: %s', score[0])
    log.info('Test accuracy: %s', score[1])

    evaluate_model(model, test_data_generator, cfg.misc.logs_path)
    manually_evaluate_model(cfg, model, test_data_generator.item_paths, cfg.dataset.img_size)
