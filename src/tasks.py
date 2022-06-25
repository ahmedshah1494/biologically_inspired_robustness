from copy import deepcopy
from typing import List, Type, Union
from mllib.datasets.dataset_factory import ImageDatasetFactory, SupportedDatasets
from mllib.models.base_models import MLPClassifier, MLP
from mllib.optimizers.configs import SGDOptimizerConfig, ReduceLROnPlateauConfig, AdamOptimizerConfig
from mllib.param import BaseParameters
from mllib.runners.configs import BaseExperimentConfig
from mllib.tasks.base_tasks import AbstractTask
from mllib.adversarial.attacks import AttackParamFactory, SupportedAttacks, SupportedBackend
import torch
import torchvision
from positional_encodings.torch_encodings import PositionalEncodingPermute2D

from models import CommonModelParams, ConsistentActivationClassifier, ConsistentActivationLayer, ConvEncoder, EyeModel, GeneralClassifier, IdentityLayer, LearnablePositionEmbedding, PositionAwareScanningConsistentActivationLayer, ScanningConsistentActivationLayer, SequentialLayers
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
    p.max_num_train = num_train
    p.max_num_test = num_test
    return p

def get_consistent_act_classifier_params(num_classes):
    cp: ConsistentActivationClassifier.ModelParams = ConsistentActivationClassifier.get_params()
    cp.common_params.num_units = num_classes
    cp.classification_params.num_classes = num_classes

def set_common_params(p, input_size: Union[int, List[int]], num_units: Union[int, List[int]], 
                        activation: Type[torch.nn.Module]=torch.nn.ReLU, bias: bool=True, dropout_p: float=0.):
    p.common_params.input_size = input_size
    p.common_params.num_units = num_units
    p.common_params.activation = activation
    p.common_params.bias = bias
    p.common_params.dropout_p = dropout_p

def set_consistency_opt_params(p, input_act_consistency, lateral_dependence_type, act_opt_step_size, 
                                max_train_time_steps, max_test_time_steps, backward_dependence_type='Linear',
                                activate_logits=True):
    p.consistency_optimization_params.act_opt_step_size = act_opt_step_size
    p.consistency_optimization_params.max_train_time_steps = max_train_time_steps
    p.consistency_optimization_params.max_test_time_steps = max_test_time_steps
    p.consistency_optimization_params.input_act_consistency = input_act_consistency
    p.consistency_optimization_params.lateral_dependence_type = lateral_dependence_type
    p.consistency_optimization_params.backward_dependence_type = backward_dependence_type
    p.consistency_optimization_params.activate_logits = activate_logits

def set_scanning_consistency_opt_params(p, kernel_size, padding, stride, 
                                            act_opt_kernel_size, act_opt_stride, 
                                            window_input_act_consistency):
    p.scanning_consistency_optimization_params.kernel_size = kernel_size
    p.scanning_consistency_optimization_params.padding = padding
    p.scanning_consistency_optimization_params.stride = stride
    p.scanning_consistency_optimization_params.act_opt_kernel_size = act_opt_kernel_size
    p.scanning_consistency_optimization_params.act_opt_stride = act_opt_stride
    p.scanning_consistency_optimization_params.window_input_act_consistency = window_input_act_consistency

def get_cifar10_adv_experiment_params(task):
    p = ConsistentActivationAdversarialExperimentConfig()
    set_SGD_params(p)
    p.optimizer_config.lr = 0.05
    set_common_training_params(p)
    test_eps = [0.0, 0.008, 0.016, 0.024, 0.032, 0.048, 0.064]
    set_adv_params(p, test_eps)
    dsp = task.get_dataset_params()
    p.act_opt_config.act_opt_lr_warmup_schedule = ActivityOptimizationSchedule.GEOM
    p.exp_name = f'{dsp.max_num_train//1000}K'
    return p

def set_scanning_consistent_activation_layer_params(p: ScanningConsistentActivationLayer.ModelParams,
                                                    num_units, input_act_opt, lat_dep_type, act_opt_lr, num_steps, kernel_size,
                                                    padding, stride, act_opt_kernel_size, act_opt_stride, activation, dropout_p, 
                                                    back_dep_type='Linear', activate_logits=True):
    p.common_params.activation = activation
    p.common_params.dropout_p = dropout_p
    p.common_params.num_units = num_units
    p.common_params.bias = True
    set_consistency_opt_params(p, input_act_opt, lat_dep_type, act_opt_lr, num_steps, num_steps, backward_dependence_type=back_dep_type, activate_logits=activate_logits)
    set_scanning_consistency_opt_params(p, kernel_size, padding, stride, act_opt_kernel_size, act_opt_stride, True)
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
        p.consistency_optimization_params.activate_logits = False
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
    num_units = 64
    act_opt_lr = 0.21
    num_steps = 16
    lat_dep = 'ReLU'

    def get_dataset_params(self) -> BaseParameters:
       return get_cifar10_params()

    def get_model_params(self) -> BaseParameters:
        fp: SequentialLayers.ModelParams = SequentialLayers.get_params()
        fp.common_params.input_size = [3, 32, 32]
        activate_logits = True

        p1: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        set_scanning_consistent_activation_layer_params(p1, self.num_units, True, self.lat_dep, self.act_opt_lr, self.num_steps, 5, 0, 3, 5, 5, torch.nn.ReLU, 0.2, activate_logits=activate_logits)

        p2: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        set_scanning_consistent_activation_layer_params(p2, self.num_units, True, self.lat_dep, self.act_opt_lr, self.num_steps, 3, 1, 1, 5, 5, torch.nn.ReLU, 0.2, activate_logits=activate_logits)

        p3: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        set_scanning_consistent_activation_layer_params(p3, self.num_units, True, self.lat_dep, self.act_opt_lr, self.num_steps, 3, 1, 2, 5, 5, torch.nn.ReLU, 0.2, activate_logits=activate_logits)
        
        fp.layer_params = [p1, p2, p3]

        cp: ConsistentActivationClassifier.ModelParams = ConsistentActivationClassifier.get_params()
        cp.common_params.num_units = 10
        cp.consistency_optimization_params.act_opt_step_size = 0.14
        cp.consistency_optimization_params.max_train_time_steps = 0
        cp.consistency_optimization_params.max_test_time_steps = 0
        cp.consistency_optimization_params.lateral_dependence_type = 'ReLU'
        cp.consistency_optimization_params.activate_logits = False
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

class Cifar10ConvConsistentActivation4LTask(Cifar10ConvConsistentActivation3LTask):
    # def get_dataset_params(self) -> BaseParameters:
    #     p = get_cifar10_params(50_000)
    #     return p

    def get_model_params(self) -> BaseParameters:
        p: GeneralClassifier.ModelParams = super().get_model_params()
        p4: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        set_scanning_consistent_activation_layer_params(p4, self.num_units, True, self.lat_dep, self.act_opt_lr, self.num_steps, 3, 1, 1, 5, 5, torch.nn.ReLU, 0.2, activate_logits=True)
        p.feature_model_params.layer_params.append(p4)
        return p

class Cifar10ConvConsistentActivation4LAutoAugmentTask(Cifar10ConvConsistentActivation4LTask):
    def get_dataset_params(self) -> BaseParameters:
        p = super().get_dataset_params()
        p.custom_transforms = (
            torchvision.transforms.Compose([
                torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.CIFAR10),
                torchvision.transforms.ToTensor()
            ]),
            torchvision.transforms.ToTensor()
        )
        return p

class Cifar10ConvConsistentActivation4LwNoActOpt512UVCortexTask(Cifar10ConvConsistentActivation4LTask):
    def get_model_params(self) -> BaseParameters:
        p: GeneralClassifier.ModelParams = super().get_model_params()
        p = Cifar10EyeModelTaskwVCortex.add_wVCtx_classifier_params(p, 0, 1e-10, False, 512)
        return p

class Cifar10ConvConsistentActivation4L64x2UMLPVCortexTask(Cifar10ConvConsistentActivation4LTask):
    def get_model_params(self) -> BaseParameters:
        p: GeneralClassifier.ModelParams = super().get_model_params()
        # p = Cifar10EyeModelTaskwVCortex.add_wVCtx_classifier_params(p, 0, 1e-10, False, 512)
        cp: MLPClassifier.ModelParams = MLPClassifier.get_params()
        cp.widths = [64, 64]
        cp.output_size = 10
        cp.dropout_p = 0.2
        p.classifier_params = cp
        return p

class Cifar10ConvConsistentActivation4Lw512UVCortexTask(Cifar10ConvConsistentActivation4LTask):
    def get_model_params(self) -> BaseParameters:
        p: GeneralClassifier.ModelParams = super().get_model_params()
        p = Cifar10EyeModelTaskwVCortex.add_wVCtx_classifier_params(p, 16, 0.21, True, 512)
        return p

class Cifar10ConvConsistentActivation5LTask(Cifar10ConvConsistentActivation4LTask):
    def get_model_params(self) -> BaseParameters:
        p: GeneralClassifier.ModelParams = super().get_model_params()
        p5: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        set_scanning_consistent_activation_layer_params(p5, self.num_units, True, self.lat_dep, self.act_opt_lr, self.num_steps, 3, 1, 2, 3, 3, torch.nn.ReLU, 0.2, activate_logits=True)
        p.feature_model_params.layer_params.append(p5)
        return p
    
    def get_experiment_params(self) -> ConsistentActivationAdversarialExperimentConfig:
        p = super().get_experiment_params()
        p.act_opt_config.num_warmup_epochs = 10
        p.act_opt_config.init_act_opt_lr = 0.001
        return p

class Cifar10ConvConsistentActivation4to5LTask(Cifar10ConvConsistentActivation5LTask):
    pass

class Cifar10ConvConsistentActivation6LTask(Cifar10ConvConsistentActivation5LTask):
    def get_model_params(self) -> BaseParameters:
        p: GeneralClassifier.ModelParams = super().get_model_params()
        p6: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        set_scanning_consistent_activation_layer_params(p6, self.num_units, True, self.lat_dep, self.act_opt_lr, self.num_steps, 3, 1, 1, 3, 3, torch.nn.ReLU, 0.2, activate_logits=True)
        p.feature_model_params.layer_params.append(p6)
        return p

class Cifar10PositionAwareConvConsistentActivation4LTask(AbstractTask):
    num_units = 64
    act_opt_lr = 0.21
    num_steps = 16
    lat_dep = 'ReLU'
    cat_pos_emb = False
    # pos_emb_cls = PositionalEncodingPermute2D
    pos_emb_cls = LearnablePositionEmbedding
    def get_dataset_params(self) -> BaseParameters:
       return get_cifar10_params()

    def get_model_params(self) -> BaseParameters:
        fp: SequentialLayers.ModelParams = SequentialLayers.get_params()
        fp.common_params.input_size = [3, 32, 32]
        activate_logits = True

        p1: PositionAwareScanningConsistentActivationLayer.ModelParams = PositionAwareScanningConsistentActivationLayer.get_params()
        p1.position_embedding_params.pos_emb_cls = self.pos_emb_cls
        p1.position_embedding_params.cat_emb = self.cat_pos_emb
        set_scanning_consistent_activation_layer_params(p1, self.num_units, True, self.lat_dep, self.act_opt_lr, self.num_steps, 5, 0, 3, 5, 5, torch.nn.ReLU, 0.2, activate_logits=activate_logits)

        p2: PositionAwareScanningConsistentActivationLayer.ModelParams = PositionAwareScanningConsistentActivationLayer.get_params()
        p2.position_embedding_params.pos_emb_cls = self.pos_emb_cls
        p2.position_embedding_params.cat_emb = self.cat_pos_emb
        set_scanning_consistent_activation_layer_params(p2, self.num_units, True, self.lat_dep, self.act_opt_lr, self.num_steps, 3, 1, 1, 5, 5, torch.nn.ReLU, 0.2, activate_logits=activate_logits)

        p3: PositionAwareScanningConsistentActivationLayer.ModelParams = PositionAwareScanningConsistentActivationLayer.get_params()
        p3.position_embedding_params.pos_emb_cls = self.pos_emb_cls
        p3.position_embedding_params.cat_emb = self.cat_pos_emb
        set_scanning_consistent_activation_layer_params(p3, self.num_units, True, self.lat_dep, self.act_opt_lr, self.num_steps, 3, 1, 2, 5, 5, torch.nn.ReLU, 0.2, activate_logits=activate_logits)
        
        p4: PositionAwareScanningConsistentActivationLayer.ModelParams = PositionAwareScanningConsistentActivationLayer.get_params()
        p4.position_embedding_params.pos_emb_cls = self.pos_emb_cls
        p4.position_embedding_params.cat_emb = self.cat_pos_emb
        set_scanning_consistent_activation_layer_params(p4, self.num_units, True, self.lat_dep, self.act_opt_lr, self.num_steps, 3, 1, 1, 5, 5, torch.nn.ReLU, 0.2, activate_logits=True)

        fp.layer_params = [p1, p2, p3, p4]

        cp: ConsistentActivationClassifier.ModelParams = ConsistentActivationClassifier.get_params()
        cp.common_params.num_units = 10
        cp.consistency_optimization_params.max_train_time_steps = 0
        cp.consistency_optimization_params.max_test_time_steps = 0
        cp.consistency_optimization_params.lateral_dependence_type = 'ReLU'
        cp.consistency_optimization_params.activate_logits = False
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
        p.exp_name = ''
        if self.num_steps == 0:
            p.exp_name += '0steps-'
        if issubclass(self.pos_emb_cls, LearnablePositionEmbedding):
            p.exp_name += 'Learned_PE-'
        if self.cat_pos_emb:
            p.exp_name += 'CatEmb-'
        p.exp_name += f'{dsp.max_num_train//1000}K'
        return p
class Cifar10ConvConsistentActivation3LwPRandHCellsTask(Cifar10ConvConsistentActivation3LTask):

    def get_model_params(self) -> BaseParameters:
        p = super().get_model_params()

        # Photoreceptors + Horizontal cells
        p0: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        p0.common_params.activation = torch.nn.ReLU
        p0.common_params.dropout_p = 0.2
        p0.scanning_consistency_optimization_params.use_forward = False
        p0.common_params.num_units = 1
        p0.common_params.bias = True
        set_consistency_opt_params(p0, True, 'ReLU', 0.14, 32, 32)
        set_scanning_consistency_opt_params(p0, 1, 0, 1, 4, 4, True)
        p0.consistency_optimization_params.sparsify_act = False
        p0.consistency_optimization_params.sparsity_coeff = 5e-4

        p1: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        p1.common_params.activation = torch.nn.ReLU
        p1.common_params.dropout_p = 0.2
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
        p0.common_params.activation = torch.nn.ReLU
        p0.common_params.dropout_p = 0.2
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

class Cifar10Conv4LTask(AbstractTask):
    def get_dataset_params(self) -> BaseParameters:
       return get_cifar10_params()

    def get_model_params(self) -> BaseParameters:
        p1: ConvEncoder.ModelParams = ConvEncoder.get_params()
        p1.common_params.input_size = [3, 32, 32]
        p1.common_params.activation = torch.nn.ReLU
        p1.common_params.num_units = [64, 64, 64, 64]
        p1.common_params.dropout_p = 0.2
        p1.conv_params.kernel_sizes = [5, 3, 3, 3]
        p1.conv_params.padding = [0, 1, 1, 1]
        p1.conv_params.strides = [3, 1, 2, 1]

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

class Cifar10Conv3Lto4LTask(Cifar10Conv4LTask):
    pass

class Cifar10EyeModelTask(AbstractTask):
    def get_dataset_params(self) -> BaseParameters:
       p = get_cifar10_params()
       return p

    @classmethod
    def get_horizontal_cell_params(cls, num_steps):
        h_params: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        set_common_params(h_params, 0, 1, activation=torch.nn.ReLU, bias=True)
        h_params.scanning_consistency_optimization_params.use_forward = False
        set_consistency_opt_params(h_params, False, 'ReLU', 0.14, num_steps, num_steps)
        set_scanning_consistency_opt_params(h_params, 1, 0, 1, 4, 4, True)
        return h_params
    
    @classmethod
    def get_bipolar_cell_params(cls, num_steps, act_opt_lr, input_act_opt, num_units,
                                    act_opt_kernel_size=5):
        bp_params: SequentialLayers.ModelParams = SequentialLayers.get_params()

        p1: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        set_common_params(p1, 0, num_units, activation=torch.nn.ReLU, bias=True, dropout_p=0.2)
        set_consistency_opt_params(p1, input_act_opt, 'ReLU', act_opt_lr, num_steps, num_steps)
        set_scanning_consistency_opt_params(p1, 5, 0, 3, act_opt_kernel_size, act_opt_kernel_size, True)

        p2 = deepcopy(p1)
        p2.scanning_consistency_optimization_params.kernel_size = 3
        p2.scanning_consistency_optimization_params.padding = 1
        p2.scanning_consistency_optimization_params.stride = 1

        p3 = deepcopy(p1)
        p3.scanning_consistency_optimization_params.kernel_size = 3
        p3.scanning_consistency_optimization_params.padding = 1
        p3.scanning_consistency_optimization_params.stride = 2

        bp_params.layer_params = [p1, p2, p3]

        return bp_params

    @classmethod
    def get_amacrine_cell_params(cls, num_steps, act_opt_lr, input_act_opt, num_units,
                                    act_opt_kernel_size=5):
        p1: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        set_common_params(p1, 0, num_units, activation=torch.nn.ReLU, bias=True, dropout_p=0.2)
        set_consistency_opt_params(p1, input_act_opt, 'ReLU', act_opt_lr, num_steps, num_steps)
        set_scanning_consistency_opt_params(p1, 3, 1, 1, act_opt_kernel_size, act_opt_kernel_size, True)
        return p1

    @classmethod
    def get_ganglion_cell_params(cls, num_steps, act_opt_lr, input_act_opt, num_units,
                                    act_opt_kernel_size=5):
        p1: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        set_common_params(p1, 0, num_units, activation=torch.nn.ReLU, bias=True, dropout_p=0.2)
        set_consistency_opt_params(p1, input_act_opt, 'ReLU', act_opt_lr, num_steps, num_steps)
        set_scanning_consistency_opt_params(p1, 3, 1, 1, act_opt_kernel_size, act_opt_kernel_size, True)
        return p1

    @classmethod
    def get_classifier_params(cl, feature_model_params, act_opt_lr, input_act_opt):
        cp: ConsistentActivationLayer = ConsistentActivationLayer.get_params()
        cp.common_params.num_units = 10
        cp.common_params.activation = torch.nn.Identity
        set_consistency_opt_params(cp, input_act_opt, 'Linear', act_opt_lr, 0, 0, activate_logits=False)

        p: GeneralClassifier.ModelParams = GeneralClassifier.get_params()
        p.feature_model_params = feature_model_params
        p.classifier_params = cp

        return p

    @classmethod
    def get_eye_model_classifier_params(cls, num_steps, act_opt_lr, input_act_opt, num_units,
                                            act_opt_kernel_size=5):
        eye_params: EyeModel.ModelParams = EyeModel.get_params()
        
        eye_params.common_params = CommonModelParams()
        eye_params.common_params.input_size = [3, 32, 32]
        eye_params.common_params.activation = torch.nn.ReLU
        eye_params.common_params.dropout_p = 0.2

        eye_params.photoreceptor_params = IdentityLayer.get_params()

        eye_params.horizontal_cell_params = Cifar10EyeModelTask.get_horizontal_cell_params(num_steps)
        eye_params.bipolar_cell_params = Cifar10EyeModelTask.get_bipolar_cell_params(num_steps, act_opt_lr, input_act_opt, num_units, act_opt_kernel_size)
        eye_params.amacrine_cell_params = Cifar10EyeModelTask.get_amacrine_cell_params(num_steps, act_opt_lr, input_act_opt, num_units, act_opt_kernel_size)
        eye_params.ganglion_cell_params = Cifar10EyeModelTask.get_ganglion_cell_params(num_steps, act_opt_lr, input_act_opt, num_units, act_opt_kernel_size)
        
        p = Cifar10EyeModelTask.get_classifier_params(eye_params, act_opt_lr, input_act_opt)

        return p

    def get_model_params(self) -> BaseParameters:
        num_steps = 16
        act_opt_lr = 0.21
        input_act_opt = True
        num_units = 64
        p = Cifar10EyeModelTask.get_eye_model_classifier_params(num_steps, act_opt_lr, input_act_opt, num_units)
        return p
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        p = get_cifar10_adv_experiment_params(self)
        return p

class Cifar10EyeModelTaskwVCortex(AbstractTask):
    def get_dataset_params(self) -> BaseParameters:
       return get_cifar10_params()

    def get_experiment_params(self) -> BaseExperimentConfig:
        p = get_cifar10_adv_experiment_params(self)
        p.optimizer_config = AdamOptimizerConfig()
        return p

    @classmethod
    def add_wVCtx_classifier_params(cls, p, num_steps, act_opt_lr, input_act_opt, num_vctx_units, num_layers=2) -> BaseParameters:
        eye_params = p.feature_model_params

        v1: ConsistentActivationLayer.ModelParams = ConsistentActivationLayer.get_params()
        set_common_params(v1, 0, num_vctx_units//8, dropout_p=0.2)
        set_consistency_opt_params(v1, input_act_opt, 'ReLU', act_opt_lr, num_steps, num_steps)

        v2: ConsistentActivationLayer.ModelParams = ConsistentActivationLayer.get_params()
        set_common_params(v2, 0, num_vctx_units, dropout_p=0.2)
        set_consistency_opt_params(v2, input_act_opt, 'ReLU', act_opt_lr/2, num_steps, num_steps)

        v4: ConsistentActivationLayer.ModelParams = ConsistentActivationLayer.get_params()
        set_common_params(v4, 0, num_vctx_units, dropout_p=0.2)
        set_consistency_opt_params(v4, input_act_opt, 'ReLU', act_opt_lr, num_steps, num_steps)

        vctx_params: SequentialLayers.ModelParams = SequentialLayers.get_params()
        vctx_params.common_params.input_size = [3, 32, 32]
        vctx_params.common_params.activation = torch.nn.ReLU
        # vctx_params.common_params.dropout_p = 0.1
        vctx_params.layer_params = [eye_params, v1, v2, v4][:num_layers+1]

        p.feature_model_params = vctx_params

        return p

    def get_model_params(self) -> BaseParameters:
        num_steps = 16
        act_opt_lr = 0.21
        input_act_opt = True
        num_units = 64
        num_vctx_units = 512
        p: GeneralClassifier.ModelParams = Cifar10EyeModelTask.get_eye_model_classifier_params(num_steps, act_opt_lr, input_act_opt, num_units)
        p = Cifar10EyeModelTaskwVCortex.add_wVCtx_classifier_params(p, num_steps, act_opt_lr, input_act_opt, num_vctx_units)
        return p

class Cifar10EyeModelTaskwNoActOptVCortex(AbstractTask):
    def get_dataset_params(self) -> BaseParameters:
       return get_cifar10_params()

    def get_experiment_params(self) -> BaseExperimentConfig:
        p = get_cifar10_adv_experiment_params(self)
        return p

    def get_model_params(self) -> BaseParameters:
        num_steps = 16
        act_opt_lr = 0.21
        input_act_opt = True
        num_units = 64
        num_vctx_units = 64
        p: GeneralClassifier.ModelParams = Cifar10EyeModelTask.get_eye_model_classifier_params(num_steps, act_opt_lr, input_act_opt, num_units)
        p = Cifar10EyeModelTaskwVCortex.add_wVCtx_classifier_params(p, 0, 1e-10, input_act_opt, num_vctx_units)
        return p

class Cifar10EyeModel128UTaskwVCortex(AbstractTask):
    def get_dataset_params(self) -> BaseParameters:
       return get_cifar10_params()

    def get_experiment_params(self) -> BaseExperimentConfig:
        p = get_cifar10_adv_experiment_params(self)
        p.batch_size = 32
        p.optimizer_config.lr = 0.01
        p.training_params.nepochs = 50
        return p

    def get_model_params(self) -> BaseParameters:
        num_steps = 16
        act_opt_lr = 0.21
        input_act_opt = True
        num_units = 128
        num_vctx_units = 1024

        p: GeneralClassifier.ModelParams = Cifar10EyeModelTask.get_eye_model_classifier_params(num_steps, act_opt_lr, input_act_opt, num_units)
        p = Cifar10EyeModelTaskwVCortex.add_wVCtx_classifier_params(p, num_steps, act_opt_lr, input_act_opt, num_vctx_units)
        return p