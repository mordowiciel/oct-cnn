import datetime
from glob import glob

from generator_resolver import GeneratorResolver
from model_evaluator import ModelEvaluator
from model_resolver import ModelResolver
from model_trainer import ModelTrainer
from oct_config import OCTConfig
from oct_logger import OCTLogger


def count_images(dir_path):
    return len(glob('{}//**//*.jpeg'.format(dir_path), recursive=True))


RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

if __name__ == '__main__':

    # Setup config
    cfg = OCTConfig('../config.ini')
    # print_cfg(cfg)

    # Setup metrics collection by wandb
    # wandb_run_name = '{}-{}-{}-{}'.format(cfg.network.architecture, RUN_TIMESTAMP, cfg.network.loss_function,
    #                                       cfg.network.optimizer)
    # wandb.init(name=wandb_run_name)

    # Setup logger
    # log = __setup_python_logger(cfg, RUN_TIMESTAMP)

    oct_logger = OCTLogger(cfg, RUN_TIMESTAMP)
    oct_logger.print_cfg()

    generator_resolver = GeneratorResolver(cfg)
    training_generator, test_generator = generator_resolver.resolve_generators()

    model_resolver = ModelResolver(cfg)
    model = model_resolver.resolve_model()

    model_trainer = ModelTrainer(cfg, model, training_generator, RUN_TIMESTAMP)
    model_trainer.train_model()

    # log.info('Fitting model...')
    # batch_history = BatchHistory(granularity=100)
    # time_history = TimeHistory()
    # training_steps_per_epoch = count_images(cfg.dataset.training_dataset_path) // cfg.training.training_batch_size
    # log.info('Training steps per epoch: %s', training_steps_per_epoch)
    #
    # history = model.fit_generator(generator=training_generator,
    #                               use_multiprocessing=False,
    #                               epochs=cfg.training.epochs,
    #                               callbacks=[batch_history, time_history],
    #                               steps_per_epoch=training_steps_per_epoch)
    #
    # log.info('Model training complete.')
    # log.info('Saving loss to batch graph.')
    # log.info('EPOCH TRAINING TIMES: %s', time_history.epochs_training_duration)
    # save_mse_to_epoch_graph(history, cfg.misc.logs_path)
    # save_loss_to_batch_graph(batch_history.history['batch'],
    #                          batch_history.history['loss'],
    #                          cfg.misc.logs_path)
    #
    # model_path = '{}//{}-{}-{}-{}.h5'.format(cfg.misc.models_path,
    #                                          cfg.network.architecture,
    #                                          RUN_TIMESTAMP,
    #                                          cfg.network.loss_function,
    #                                          cfg.network.optimizer)
    # log.info('Saving model at path %s', model_path)
    # model.save(model_path)

    model_evaluator = ModelEvaluator(cfg, model, test_generator)
    model_evaluator.evaluate_model()



    # test_image_count = count_images(cfg.dataset.test_dataset_path)
    # log.info('Image count %s', test_image_count)
    # test_steps_per_epoch = test_image_count // cfg.training.test_batch_size
    # log.info('Test steps per epoch %s', test_steps_per_epoch)
    #
    # score = model.evaluate_generator(test_generator, steps=test_steps_per_epoch, use_multiprocessing=False,
    #                                  verbose=0)
    #
    # log.info('Training evaluation:')
    # log.info('Test loss: %s', score[0])
    # log.info('Test accuracy: %s', score[1])
    #
    # evaluate_model(model, test_generator, cfg.dataset.test_dataset_path, test_steps_per_epoch, cfg.misc.logs_path)
    # manually_evaluate_model(cfg, model, cfg.dataset.test_dataset_path, cfg.dataset.img_size)
