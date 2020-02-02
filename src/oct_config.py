import configparser
from ast import literal_eval


class OCTConfig:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.dataset = DatasetConfig(config_file_path)
        self.training = TrainingConfig(config_file_path)
        self.network = NetworkConfig(config_file_path)
        self.augmentation = AugmentationConfig(config_file_path)
        self.misc = MiscConfig(config_file_path)


class TrainingConfig:
    def __init__(self, config_file_path):
        config = configparser.ConfigParser()
        config.read(config_file_path)

        training_config = config['training']
        self.epochs = int(training_config['epochs'])
        self.training_batch_size = int(training_config['training_batch_size'])
        self.test_batch_size = int(training_config['test_batch_size'])
        self.early_stopping_monitor = str(training_config['early_stopping_monitor'])
        self.early_stopping_patience = int(training_config['early_stopping_patience'])
        self.early_stopping_min_delta = float(training_config['early_stopping_min_delta'])
        self.early_stopping_baseline = literal_eval(training_config['early_stopping_baseline'])


class NetworkConfig:
    def __init__(self, config_file_path):
        config = configparser.ConfigParser()
        config.read(config_file_path)

        network_config = config['network']
        self.architecture = str(network_config['architecture'])
        self.loss_function = str(network_config['loss_function'])
        self.optimizer = str(network_config['optimizer'])


class DatasetConfig:
    def __init__(self, config_file_path):
        config = configparser.ConfigParser()
        config.read(config_file_path)

        dataset_config = config['dataset']
        self.img_size = literal_eval(dataset_config['img_size'])
        self.input_shape = literal_eval(dataset_config['input_shape'])
        self.training_dataset_path = str(dataset_config['training_dataset_path'])
        self.test_dataset_path = str(dataset_config['test_dataset_path'])
        self.validation_split = float(dataset_config['validation_split'])
        self.generate_extended_test_dataset = dataset_config['generate_extended_test_dataset'] == "True"


class AugmentationConfig:
    def __init__(self, config_file_path):
        config = configparser.ConfigParser()
        config.read(config_file_path)

        augmentation_config = config['augmentation']
        self.use_data_augmentation = augmentation_config['use_data_augmentation'] == "True"
        self.horizontal_flip = augmentation_config['horizontal_flip'] == "True"
        self.width_shift_range = float(augmentation_config['width_shift_range'])
        self.height_shift_range = float(augmentation_config['height_shift_range'])
        self.brightness_range = literal_eval(augmentation_config['brightness_range'])
        self.contrast_range = literal_eval(augmentation_config['contrast_range'])
        self.gaussian_noise_var_range = literal_eval(augmentation_config['gaussian_noise_var_range'])
        self.preprocessing_functions = literal_eval(augmentation_config['preprocessing_functions'])
        self.dtype = str(augmentation_config['dtype'])


class MiscConfig:
    def __init__(self, config_file_path):
        config = configparser.ConfigParser()
        config.read(config_file_path)

        misc_config = config['misc']
        self.models_path = str(misc_config['models_path'])
        self.logs_path = str(misc_config['logs_path'])
