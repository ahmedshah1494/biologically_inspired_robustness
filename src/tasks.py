from copy import deepcopy
from mllib.datasets.dataset_factory import ImageDatasetFactory, SupportedDatasets
from mllib.models.base_models import MLPClassifier, MLP
from mllib.optimizers.configs import SGDOptimizerConfig, ReduceLROnPlateauConfig
from mllib.param import BaseParameters
from mllib.runners.configs import BaseExperimentConfig
from mllib.tasks.base_tasks import AbstractTask
from mllib.adversarial.attacks import AttackParamFactory, SupportedAttacks, SupportedBackend
import torch
import torchvision

from models import ConsistentActivationClassifier, GeneralClassifier, ScanningConsistentActivationLayer, SequentialLayers
from runners import AdversarialExperimentConfig, ConsistentActivationAdversarialExperimentConfig
from trainers import ActivityOptimizationSchedule

def set_SGD_params(p: BaseExperimentConfig):
    p.optimizer_config = SGDOptimizerConfig()
    p.optimizer_config.lr = 0.01
    p.optimizer_config.momentum = 0.9
    p.optimizer_config.nesterov = True
    p.scheduler_config = ReduceLROnPlateauConfig()

def set_common_training_params(p: BaseExperimentConfig):
    p.batch_size = 256
    p.training_params.nepochs = 200
    p.num_trainings = 10
    p.logdir = '../logs'
    p.training_params.early_stop_patience = 20

def set_adv_params(p: AdversarialExperimentConfig, test_eps):
    p.adv_config.training_attack_params = None
    def eps_to_attack(eps):
        atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.PGDLINF, SupportedBackend.TORCHATTACKS)
        atk_p.eps = eps
        atk_p.nsteps = 50
        atk_p.step_size = eps/40
        atk_p.random_start = True
        return atk_p
    p.adv_config.testing_attack_params = [eps_to_attack(eps) for eps in test_eps]

def get_cifar10_params(num_train=25000, num_test=1000):
    p = ImageDatasetFactory.get_params()
    p.dataset = SupportedDatasets.CIFAR10
    p.datafolder = '/home/mshah1/workhorse3/'
    p.max_num_train = num_train
    p.max_num_test = num_test
    return p
class MNISTMLP(AbstractTask):
    def get_dataset_params(self) -> BaseParameters:
        return get_cifar10_params()
    
    def get_model_params(self) -> BaseParameters:
        p = MLPClassifier.get_params()
        p.widths = [64]
        p.input_size = 28*28
        p.output_size = 10
        return p
    
    def get_experiment_params(self) -> BaseExperimentConfig:
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
        p.consistency_optimization_params.max_train_time_steps = 32
        p.consistency_optimization_params.max_test_time_steps = 32
        p.consistency_optimization_params.lateral_dependence_type = 'ReLU'
        p.classification_params.num_classes = 10
        return p
    
    def get_experiment_params(self) -> ConsistentActivationAdversarialExperimentConfig:
        p = ConsistentActivationAdversarialExperimentConfig()
        set_SGD_params(p)
        set_common_training_params(p)
        test_eps = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
        set_adv_params(p, test_eps)        
        return p

class Cifar10ConvConsistentActivation3LTask(AbstractTask):
    def get_dataset_params(self) -> BaseParameters:
       return get_cifar10_params()

    def get_model_params(self) -> BaseParameters:
        fp: SequentialLayers.ModelParams = SequentialLayers.get_params()
        fp.common_params.input_size = [3, 32, 32]
        fp.common_params.activation = torch.nn.ReLU
        fp.common_params.dropout_p = 0.3

        p1: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        p1.common_params.num_units = 64
        p1.common_params.bias = True
        p1.consistency_optimization_params.act_opt_step_size = 0.14
        p1.consistency_optimization_params.max_train_time_steps = 32
        p1.consistency_optimization_params.max_test_time_steps = 32
        p1.consistency_optimization_params.lateral_dependence_type = 'ReLU'
        p1.scanning_consistency_optimization_params.kernel_size = 5
        p1.scanning_consistency_optimization_params.padding = 0
        p1.scanning_consistency_optimization_params.stride = 3
        p1.scanning_consistency_optimization_params.act_opt_kernel_size = 5
        p1.scanning_consistency_optimization_params.act_opt_stride = 5
        p1.scanning_consistency_optimization_params.window_input_act_consistency = True

        p2 = deepcopy(p1)
        p2.scanning_consistency_optimization_params.kernel_size = 3
        p2.scanning_consistency_optimization_params.padding = 1
        p2.scanning_consistency_optimization_params.stride = 2

        p3 = deepcopy(p1)
        p3.scanning_consistency_optimization_params.kernel_size = 3
        p3.scanning_consistency_optimization_params.padding = 1
        p3.scanning_consistency_optimization_params.stride = 1
        
        fp.layer_params = [p1, p2, p3]

        cp: ConsistentActivationClassifier.ModelParams = ConsistentActivationClassifier.get_params()
        cp.common_params.num_units = 10
        cp.consistency_optimization_params.act_opt_step_size = 0.14
        cp.consistency_optimization_params.max_train_time_steps = 0
        cp.consistency_optimization_params.max_test_time_steps = 0
        cp.consistency_optimization_params.lateral_dependence_type = 'ReLU'
        cp.classification_params.num_classes = 10

        p: GeneralClassifier.ModelParams = GeneralClassifier.get_params()
        p.feature_model_params = fp
        p.classifier_params = cp

        return p

    def get_experiment_params(self) -> ConsistentActivationAdversarialExperimentConfig:
        p = ConsistentActivationAdversarialExperimentConfig()
        set_SGD_params(p)
        set_common_training_params(p)
        test_eps = [0.0, 0.008, 0.016, 0.024, 0.032, 0.048, 0.064]
        set_adv_params(p, test_eps)
        p.optimizer_config.lr = 0.05
        dsp = self.get_dataset_params()
        p.act_opt_config.act_opt_lr_warmup_schedule = ActivityOptimizationSchedule.GEOM
        p.exp_name = f'{dsp.max_num_train//1000}K'
        return p