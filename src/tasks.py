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

from models import ConsistentActivationClassifier, ConvEncoder, GeneralClassifier, ScanningConsistentActivationLayer, SequentialLayers
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
    p.training_params.tracked_metric = 'val_loss'
    p.training_params.tracking_mode = 'min'

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
    # p.custom_transforms = (
    #     torchvision.transforms.Compose([
    #         torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.CIFAR10),
    #         torchvision.transforms.ToTensor()
    #     ]),
    #     torchvision.transforms.ToTensor()
    # )
    p.max_num_train = num_train
    p.max_num_test = num_test
    return p

def get_consistent_act_classifier_params(num_classes):
    cp: ConsistentActivationClassifier.ModelParams = ConsistentActivationClassifier.get_params()
    cp.common_params.num_units = num_classes
    cp.classification_params.num_classes = num_classes

def set_consistency_opt_params(p, input_act_consistency, lateral_dependence_type, act_opt_step_size, max_train_time_steps, max_test_time_steps):
    p.consistency_optimization_params.act_opt_step_size = act_opt_step_size
    p.consistency_optimization_params.max_train_time_steps = max_train_time_steps
    p.consistency_optimization_params.max_test_time_steps = max_test_time_steps
    p.consistency_optimization_params.input_act_consistency = input_act_consistency
    p.consistency_optimization_params.lateral_dependence_type = lateral_dependence_type

def set_scanning_consistency_opt_params(p, kernel_size, padding, stride, 
                                            act_opt_kernel_size, act_opt_stride, 
                                            window_input_act_consistency):
    p.scanning_consistency_optimization_params.kernel_size = kernel_size
    p.scanning_consistency_optimization_params.padding = padding
    p.scanning_consistency_optimization_params.stride = stride
    p.scanning_consistency_optimization_params.act_opt_kernel_size = act_opt_kernel_size
    p.scanning_consistency_optimization_params.act_opt_stride = act_opt_stride
    p.scanning_consistency_optimization_params.window_input_act_consistency = window_input_act_consistency
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
        p.consistency_optimization_params.input_act_consistency = True
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
        fp.common_params.dropout_p = 0.2

        p1: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        p1.common_params.num_units = 64
        p1.common_params.bias = True
        set_consistency_opt_params(p1, True, 'ReLU', 0.14, 32, 32)
        set_scanning_consistency_opt_params(p1, 5, 0, 3, 5, 5, True)

        p2 = deepcopy(p1)
        p2.scanning_consistency_optimization_params.kernel_size = 3
        p2.scanning_consistency_optimization_params.padding = 1
        p2.scanning_consistency_optimization_params.stride = 1

        p3 = deepcopy(p1)
        p3.scanning_consistency_optimization_params.kernel_size = 3
        p3.scanning_consistency_optimization_params.padding = 1
        p3.scanning_consistency_optimization_params.stride = 2
        
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
        p.optimizer_config.lr = 0.05
        set_common_training_params(p)
        test_eps = [0.0, 0.008, 0.016, 0.024, 0.032, 0.048, 0.064]
        set_adv_params(p, test_eps)
        dsp = self.get_dataset_params()
        p.act_opt_config.act_opt_lr_warmup_schedule = ActivityOptimizationSchedule.GEOM
        p.exp_name = f'{dsp.max_num_train//1000}K'
        return p

class Cifar10ConvConsistentActivation3LwPRandHCellsTask(Cifar10ConvConsistentActivation3LTask):

    def get_model_params(self) -> BaseParameters:
        p = super().get_model_params()

        # Photoreceptors + Horizontal cells
        p0: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        p0.scanning_consistency_optimization_params.use_forward = False
        p0.common_params.num_units = 1
        p0.common_params.bias = True
        set_consistency_opt_params(p0, True, 'ReLU', 0.14, 32, 32)
        set_scanning_consistency_opt_params(p0, 1, 0, 1, 4, 4, True)
        p0.consistency_optimization_params.sparsify_act = False
        p0.consistency_optimization_params.sparsity_coeff = 5e-4

        p1: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        # p1.scanning_consistency_optimization_params.use_forward = False
        p1.common_params.num_units = 64
        p1.common_params.bias = True
        set_consistency_opt_params(p1, True, 'ReLU', 0.14, 32, 32)
        set_scanning_consistency_opt_params(p1, 5, 0, 3, 5, 5, True)

        p2 = deepcopy(p1)
        p2.scanning_consistency_optimization_params.kernel_size = 3
        p2.scanning_consistency_optimization_params.padding = 1
        p2.scanning_consistency_optimization_params.stride = 1

        p3 = deepcopy(p1)
        p3.scanning_consistency_optimization_params.kernel_size = 3
        p3.scanning_consistency_optimization_params.padding = 1
        p3.scanning_consistency_optimization_params.stride = 2

        p4 = deepcopy(p1)
        p4.scanning_consistency_optimization_params.kernel_size = 3
        p4.scanning_consistency_optimization_params.padding = 1
        p4.scanning_consistency_optimization_params.stride = 1
        
        p.feature_model_params.layer_params = [p0, p1, p2, p3]

        # p.feature_model_params.layer_params = [p1]+p.feature_model_params.layer_params
        return p

    def get_experiment_params(self) -> ConsistentActivationAdversarialExperimentConfig:
        p = ConsistentActivationAdversarialExperimentConfig()
        set_SGD_params(p)
        p.optimizer_config.lr = 0.05
        set_common_training_params(p)
        test_eps = [0.0, 0.008, 0.016, 0.024, 0.032, 0.048, 0.064]
        set_adv_params(p, test_eps)
        p.act_opt_config.act_opt_lr_warmup_schedule = ActivityOptimizationSchedule.GEOM

        dsp = self.get_dataset_params()
        mp = self.get_model_params()
        sparsify = mp.feature_model_params.layer_params[0].consistency_optimization_params.sparsify_act
        sparsity_coeff = mp.feature_model_params.layer_params[0].consistency_optimization_params.sparsity_coeff
        if sparsify:
            p.exp_name += f'-{sparsity_coeff}Sparse'
        inp_act_opt = mp.feature_model_params.layer_params[0].consistency_optimization_params.input_act_consistency
        back_dep = mp.feature_model_params.layer_params[0].consistency_optimization_params.backward_dependence_type
        if inp_act_opt:
            p.exp_name += f'-{back_dep}BackDep'
        p.exp_name += f'-{dsp.max_num_train//1000}K'
        return p

class Cifar10Conv3LwPRandHCellsTask(Cifar10ConvConsistentActivation3LwPRandHCellsTask):

    def get_model_params(self) -> BaseParameters:
        p = super().get_model_params()
        fp = p.feature_model_params
        fp.common_params.dropout_p = 0.
        
        # Photoreceptors + Horizontal cells
        p0: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        p0.scanning_consistency_optimization_params.use_forward = False
        p0.common_params.num_units = 1
        p0.common_params.bias = True
        set_consistency_opt_params(p0, False, 'ReLU', 0.14, 32, 32)
        set_scanning_consistency_opt_params(p0, 1, 0, 1, 4, 4, True)
        p0.consistency_optimization_params.sparsify_act = False
        p0.consistency_optimization_params.sparsity_coeff = 5e-4

        p1: ConvEncoder.ModelParams = ConvEncoder.get_params()
        p1.common_params.num_units = [64, 64, 64]
        p1.common_params.dropout_p = 0.2
        p1.conv_params.kernel_sizes = [5, 3, 3]
        p1.conv_params.padding = [0, 1, 1]
        p1.conv_params.strides = [3, 1, 2]
        
        p.feature_model_params.layer_params = [p0, p1]
        return p

class Cifar10Conv4LwPRandHCellsTask(Cifar10Conv3LwPRandHCellsTask):
    def get_model_params(self) -> BaseParameters:
        p = super().get_model_params()

        p1: ConvEncoder.ModelParams = ConvEncoder.get_params()
        p1.common_params.num_units = [64, 64, 64, 64]
        p1.common_params.dropout_p = 0.2
        p1.conv_params.kernel_sizes = [5, 3, 3, 3]
        p1.conv_params.padding = [0, 1, 1, 1]
        p1.conv_params.strides = [3, 1, 2, 1]
        
        p.feature_model_params.layer_params[1] = p1
        return p

class Cifar10Conv3LTask(AbstractTask):
    def get_dataset_params(self) -> BaseParameters:
       return get_cifar10_params()

    def get_model_params(self) -> BaseParameters:
        p1: ConvEncoder.ModelParams = ConvEncoder.get_params()
        p1.common_params.input_size = [3, 32, 32]
        p1.common_params.activation = torch.nn.ReLU
        p1.common_params.num_units = [64, 64, 64]
        p1.common_params.dropout_p = 0.2
        p1.conv_params.kernel_sizes = [5, 3, 3]
        p1.conv_params.padding = [0, 1, 1]
        p1.conv_params.strides = [3, 1, 2]

        cp: ConsistentActivationClassifier.ModelParams = ConsistentActivationClassifier.get_params()
        cp.common_params.num_units = 10
        cp.consistency_optimization_params.act_opt_step_size = 0.14
        cp.consistency_optimization_params.max_train_time_steps = 0
        cp.consistency_optimization_params.max_test_time_steps = 0
        cp.consistency_optimization_params.lateral_dependence_type = 'ReLU'
        cp.classification_params.num_classes = 10

        p: GeneralClassifier.ModelParams = GeneralClassifier.get_params()
        p.feature_model_params = p1
        p.classifier_params = cp

        return p

    def get_experiment_params(self) -> ConsistentActivationAdversarialExperimentConfig:
        p = ConsistentActivationAdversarialExperimentConfig()
        set_SGD_params(p)
        p.optimizer_config.lr = 0.05
        set_common_training_params(p)
        test_eps = [0.0, 0.008, 0.016, 0.024, 0.032, 0.048, 0.064]
        set_adv_params(p, test_eps)
        dsp = self.get_dataset_params()
        p.act_opt_config.act_opt_lr_warmup_schedule = ActivityOptimizationSchedule.GEOM
        p.exp_name = f'{dsp.max_num_train//1000}K'
        return p