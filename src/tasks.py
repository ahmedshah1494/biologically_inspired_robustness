from mllib.datasets.dataset_factory import ImageDatasetFactory, SupportedDatasets
from mllib.models.base_models import MLPClassifier
from mllib.optimizers.configs import SGDOptimizerConfig, ReduceLROnPlateauConfig
from mllib.param import BaseParameters
from mllib.runners.configs import BaseExperimentConfig
from mllib.tasks.base_tasks import AbstractTask

from models import ConsistentActivationClassifier

class MNISTConsistentActivationClassifier(AbstractTask):
    def get_dataset_params(self) -> BaseParameters:
        p = ImageDatasetFactory.get_params()
        p.dataset = SupportedDatasets.MNIST
        p.datafolder = '.'
        p.max_num_train = 1000
        p.max_num_test = 1000
        return p
    
    def get_model_params(self) -> BaseParameters:
        p: ConsistentActivationClassifier.ModelParams = ConsistentActivationClassifier.get_params()
        p.common_params.input_size = 28*28
        p.common_params.num_units = 64
        p.consistency_optimization_params.max_train_time_steps = 4
        p.consistency_optimization_params.max_test_time_steps = 4
        p.classification_params.num_classes = 10
        return p
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        p = BaseExperimentConfig()
        p.batch_size = 128
        p.optimizer_config = SGDOptimizerConfig()
        p.optimizer_config.lr = 0.01
        p.optimizer_config.momentum = 0.9
        p.optimizer_config.nesterov = True
        p.scheduler_config = ReduceLROnPlateauConfig()
        p.training_params.nepochs = 2
        p.logdir = 'logs'
        return p