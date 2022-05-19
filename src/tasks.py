from mllib.datasets.dataset_factory import ImageDatasetFactory, SupportedDatasets
from mllib.models.base_models import MLPClassifier
from mllib.optimizers.configs import SGDOptimizerConfig, ReduceLROnPlateauConfig
from mllib.param import BaseParameters
from mllib.runners.configs import BaseExperimentConfig
from mllib.tasks.base_tasks import AbstractTask
from mllib.adversarial.attacks import AttackParamFactory, SupportedAttacks, SupportedBackend

from models import ConsistentActivationClassifier
from runners import AdversarialExperimentConfig

class MNISTConsistentActivationClassifier(AbstractTask):
    def get_dataset_params(self) -> BaseParameters:
        p = ImageDatasetFactory.get_params()
        p.dataset = SupportedDatasets.MNIST
        p.datafolder = '/home/mshah1/workhorse3/'
        p.max_num_train = 10000
        p.max_num_test = 1000
        return p
    
    def get_model_params(self) -> BaseParameters:
        p: ConsistentActivationClassifier.ModelParams = ConsistentActivationClassifier.get_params()
        p.common_params.input_size = 28*28
        p.common_params.num_units = 64
        p.consistency_optimization_params.act_opt_step_size = 0.14
        p.consistency_optimization_params.max_train_time_steps = 0
        p.consistency_optimization_params.max_test_time_steps = 0
        p.classification_params.num_classes = 10
        return p
    
    def get_experiment_params(self) -> AdversarialExperimentConfig:
        p = AdversarialExperimentConfig()
        p.batch_size = 256
        p.optimizer_config = SGDOptimizerConfig()
        p.optimizer_config.lr = 0.01
        p.optimizer_config.momentum = 0.9
        p.optimizer_config.nesterov = True
        p.scheduler_config = ReduceLROnPlateauConfig()
        p.training_params.nepochs = 80
        p.logdir = '../logs'
        adv_config = p.adv_config
        adv_config.training_attack_params = None
        test_eps = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]        
        def eps_to_attack(eps):
            atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.PGDLINF, SupportedBackend.TORCHATTACKS)
            atk_p.eps = eps
            atk_p.nsteps = 50
            atk_p.step_size = eps/40
            atk_p.random_start = True
            return atk_p
        adv_config.testing_attack_params = [eps_to_attack(eps) for eps in test_eps]
        return p