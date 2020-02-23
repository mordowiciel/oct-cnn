import argparse
import datetime

from augmentation.augmentation_processor import AugmentationProcessor
from generator_resolver import GeneratorResolver
from model_evaluator import ModelEvaluator
from model_resolver import ModelResolver
from model_trainer import ModelTrainer
from oct_config import OCTConfig
from oct_logger import OCTLogger

if __name__ == '__main__':
    RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default='../config.ini', required=False)

    args = parser.parse_args()

    cfg = OCTConfig(args.config_path)
    oct_logger = OCTLogger(cfg, RUN_TIMESTAMP)
    oct_logger.print_cfg()

    generator_resolver = GeneratorResolver(cfg)
    # training_image_data_generator, test_image_data_generator, val_image_data_generator = generator_resolver\
    #     .provide_image_data_generators()
    training_data_iterator, test_data_iterator, val_data_iterator = generator_resolver.resolve_data_iterators()

    # f = open("val_generator_test_new.txt", "w+")
    # f.write(str(val_data_iterator.filenames))
    #
    # f = open("train_generator_test_new.txt", "w+")
    # f.write(str(training_data_iterator.filenames))

    model_resolver = ModelResolver(cfg)
    model = model_resolver.resolve_model()

    augmented_image_data_generator = generator_resolver.provide_augmented_image_data_generator()
    augmentation_processor = AugmentationProcessor(cfg, augmented_image_data_generator)
    augmentation_processor.perform_data_augmentation()

    model_trainer = ModelTrainer(cfg, model, training_data_iterator, val_data_iterator, RUN_TIMESTAMP)
    model_trainer.train_model()

    model_evaluator = ModelEvaluator(cfg, model, test_data_iterator)
    model_evaluator.evaluate_model()
