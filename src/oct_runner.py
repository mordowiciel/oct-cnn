import datetime

from generator_resolver import GeneratorResolver
from model_evaluator import ModelEvaluator
from model_resolver import ModelResolver
from model_trainer import ModelTrainer
from oct_config import OCTConfig
from oct_logger import OCTLogger

if __name__ == '__main__':
    RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    cfg = OCTConfig('../config.ini')
    oct_logger = OCTLogger(cfg, RUN_TIMESTAMP)
    oct_logger.print_cfg()

    generator_resolver = GeneratorResolver(cfg)
    training_generator, test_generator = generator_resolver.resolve_generators()

    model_resolver = ModelResolver(cfg)
    model = model_resolver.resolve_model()

    model_trainer = ModelTrainer(cfg, model, training_generator, RUN_TIMESTAMP)
    model_trainer.train_model()

    model_evaluator = ModelEvaluator(cfg, model, test_generator)
    model_evaluator.evaluate_model()