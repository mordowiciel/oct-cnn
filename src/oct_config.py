import configparser
from ast import literal_eval


class OCTConfig:
    def __init__(self, config_file_path):
        self.dataset = DatasetConfig(config_file_path)
        self.training = TrainingConfig(config_file_path)
        self.network = NetworkConfig(config_file_path)


class TrainingConfig:
    def __init__(self, config_file_path):
        config = configparser.ConfigParser()
        config.read(config_file_path)

        training_config = config['training']
        self.epochs = int(training_config['epochs'])
        self.batch_size = int(training_config['batch_size'])


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
        self.generate_extended_test_dataset = bool(dataset_config['generate_extended_test_dataset'])
