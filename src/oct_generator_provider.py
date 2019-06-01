from extended_test_data_generator import ExtendedTestDataGenerator
from oct_data_generator import OCTDataGenerator


def provide_generators(cfg):
    if cfg.dataset.generate_extended_test_dataset:
        test_data_generator = ExtendedTestDataGenerator(dataset_path=cfg.dataset.training_dataset_path,
                                                        batch_size=cfg.training.batch_size,
                                                        dim=cfg.dataset.img_size,
                                                        n_channels=1,
                                                        n_classes=4,
                                                        shuffle=True)

        paths_to_skip = test_data_generator.item_paths
        training_data_generator = OCTDataGenerator(dataset_path=cfg.dataset.training_dataset_path,
                                                   paths_to_skip=paths_to_skip,
                                                   batch_size=cfg.training.batch_size,
                                                   dim=cfg.dataset.img_size,
                                                   n_channels=1,
                                                   n_classes=4,
                                                   shuffle=True)
    else:
        test_data_generator = OCTDataGenerator(dataset_path=cfg.dataset.test_dataset_path,
                                               batch_size=cfg.training.batch_size,
                                               dim=cfg.dataset.img_size,
                                               n_channels=1,
                                               n_classes=4,
                                               shuffle=True)
        training_data_generator = OCTDataGenerator(dataset_path=cfg.dataset.training_dataset_path,
                                                   batch_size=cfg.training.batch_size,
                                                   dim=cfg.dataset.img_size,
                                                   n_channels=1,
                                                   n_classes=4,
                                                   shuffle=True)

    return training_data_generator, test_data_generator
