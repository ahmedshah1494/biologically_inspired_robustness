import torchvision
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import NormalizationLayer
from adversarialML.biologically_inspired_models.src.models import (ConsistentActivationClassifier, 
    SequentialLayers, ScanningConsistentActivationLayer, GeneralClassifier, ConvEncoder, LinearLayer,
    FlattenLayer, CommonModelParams)
from adversarialML.biologically_inspired_models.src.retina_preproc import (
    RetinaBlurFilter, RetinaNonUniformPatchEmbedding, RetinaWarp, GaussianBlurLayer)
from adversarialML.biologically_inspired_models.src.supconloss import \
    TwoCropTransform
from adversarialML.biologically_inspired_models.src.trainers import ConsistentActivationModelAdversarialTrainer, ActivityOptimizationParams
from mllib.optimizers.configs import (AdamOptimizerConfig,
                                      CosineAnnealingWarmRestartsConfig,
                                      CyclicLRConfig, LinearLRConfig,
                                      ReduceLROnPlateauConfig,
                                      SequentialLRConfig, OneCycleLRConfig, SGDOptimizerConfig)
from mllib.runners.configs import BaseExperimentConfig, TrainingParams
from mllib.tasks.base_tasks import AbstractTask
from torch import nn
from adversarialML.biologically_inspired_models.src.task_utils import *

from adversarialML.biologically_inspired_models.src.trainers import ActivityOptimizationSchedule

def set_consistency_opt_params(p, input_act_consistency, lateral_dependence_type, act_opt_step_size, 
                                max_train_time_steps, max_test_time_steps, backward_dependence_type='Linear',
                                activate_logits=True, act_opt_mask_p=0.):
    p.consistency_optimization_params.act_opt_step_size = act_opt_step_size
    p.consistency_optimization_params.max_train_time_steps = max_train_time_steps
    p.consistency_optimization_params.max_test_time_steps = max_test_time_steps
    p.consistency_optimization_params.input_act_consistency = input_act_consistency
    p.consistency_optimization_params.lateral_dependence_type = lateral_dependence_type
    p.consistency_optimization_params.backward_dependence_type = backward_dependence_type
    p.consistency_optimization_params.activate_logits = activate_logits
    p.consistency_optimization_params.act_opt_mask_p = act_opt_mask_p

def set_scanning_consistency_opt_params(p, kernel_size, padding, stride, 
                                            act_opt_kernel_size, act_opt_stride, 
                                            window_input_act_consistency):
    p.scanning_consistency_optimization_params.kernel_size = kernel_size
    p.scanning_consistency_optimization_params.padding = padding
    p.scanning_consistency_optimization_params.stride = stride
    p.scanning_consistency_optimization_params.act_opt_kernel_size = act_opt_kernel_size
    p.scanning_consistency_optimization_params.act_opt_stride = act_opt_stride
    p.scanning_consistency_optimization_params.window_input_act_consistency = window_input_act_consistency

def set_scanning_consistent_activation_layer_params(p: ScanningConsistentActivationLayer.ModelParams,
                                                    num_units, input_act_opt, lat_dep_type, act_opt_lr, num_steps, kernel_size,
                                                    padding, stride, act_opt_kernel_size, act_opt_stride, activation, dropout_p, 
                                                    back_dep_type='Linear', activate_logits=True, act_opt_mask_p=0.):
    p.common_params.activation = activation
    p.common_params.dropout_p = dropout_p
    p.common_params.num_units = num_units
    p.common_params.bias = True
    set_consistency_opt_params(p, input_act_opt, lat_dep_type, act_opt_lr, num_steps, num_steps, backward_dependence_type=back_dep_type, activate_logits=activate_logits, 
                                act_opt_mask_p=act_opt_mask_p)
    set_scanning_consistency_opt_params(p, kernel_size, padding, stride, act_opt_kernel_size, act_opt_stride, True)


class SVHNMLP1LClassifier(AbstractTask):
    num_layers = 2
    num_units = 64
    input_size = 3*32*32

    def get_dataset_params(self):
        p = get_svhn_params()
        return p
    
    def get_model_params(self):
        mlp_p = SequentialLayers.ModelParams(SequentialLayers, 
        [FlattenLayer.get_params()]+[LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.input_size, self.num_units))]*self.num_layers)
        mlp_p.common_params.input_size = [self.input_size]

        cls_p = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.num_units, 10))

        p: GeneralClassifier.ModelParams = GeneralClassifier.get_params()
        p.feature_model_params = mlp_p
        p.classifier_params = cls_p
        return p
    
    def get_experiment_params(self):
        nepochs = 60
        p = BaseExperimentConfig(
            ConsistentActivationModelAdversarialTrainer.TrainerParams(ConsistentActivationModelAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=20, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=True),
                AdversarialParams(testing_attack_params=get_apgd_inf_params([0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1], 50))
            ),
            SGDOptimizerConfig(lr=0.01, weight_decay=5e-5, momentum=0.9, nesterov=True),
            ReduceLROnPlateauConfig(),
            # OneCycleLRConfig(max_lr=1., epochs=nepochs, steps_per_epoch=176),
            logdir=LOGDIR, batch_size=256
        )
        return p

class MNISTMLP1LClassifier(AbstractTask):
    num_layers = 1
    num_units = 64

    def get_dataset_params(self):
        p = get_mnist_params()
        return p
    
    def get_model_params(self):
        mlp_p = SequentialLayers.ModelParams(SequentialLayers, 
        [FlattenLayer.get_params()]+[LinearLayer.ModelParams(LinearLayer, CommonModelParams(28*28, self.num_units))]*self.num_layers)
        mlp_p.common_params.input_size = [28*28]

        cls_p = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.num_units, 10))

        p: GeneralClassifier.ModelParams = GeneralClassifier.get_params()
        p.feature_model_params = mlp_p
        p.classifier_params = cls_p
        return p
    
    def get_experiment_params(self):
        nepochs = 60
        p = BaseExperimentConfig(
            ConsistentActivationModelAdversarialTrainer.TrainerParams(ConsistentActivationModelAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=20, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=True),
                AdversarialParams(testing_attack_params=get_apgd_inf_params([0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1], 50))
            ),
            SGDOptimizerConfig(lr=0.01, weight_decay=5e-5, momentum=0.9, nesterov=True),
            ReduceLROnPlateauConfig(),
            # OneCycleLRConfig(max_lr=1., epochs=nepochs, steps_per_epoch=176),
            logdir=LOGDIR, batch_size=256
        )
        return p

class MNISTConsistentActivationClassifier(AbstractTask):
    def get_dataset_params(self):
        p = get_mnist_params()
        return p
    
    def get_model_params(self):
        p: ConsistentActivationClassifier.ModelParams = ConsistentActivationClassifier.get_params()
        p.common_params.input_size = 28*28
        p.common_params.num_units = 64
        p.consistency_optimization_params.act_opt_step_size = 1.
        p.consistency_optimization_params.max_train_time_steps = 1
        p.consistency_optimization_params.max_test_time_steps = 1
        p.consistency_optimization_params.lateral_dependence_type = 'ReLU'
        p.consistency_optimization_params.input_act_consistency = True
        p.consistency_optimization_params.activate_logits = False
        p.classification_params.num_classes = 10
        return p
    
    def get_experiment_params(self):
        nepochs = 60
        p = BaseExperimentConfig(
            ConsistentActivationModelAdversarialTrainer.TrainerParams(ConsistentActivationModelAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=20, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=True),
                AdversarialParams(testing_attack_params=get_apgd_inf_params([0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1], 50))
            ),
            SGDOptimizerConfig(lr=0.01, weight_decay=5e-5, momentum=0.9, nesterov=True),
            ReduceLROnPlateauConfig(),
            # OneCycleLRConfig(max_lr=1., epochs=nepochs, steps_per_epoch=176),
            logdir=LOGDIR, batch_size=256
        )
        return p

class MNISTConv2LConsistentActivation8StepsTask(AbstractTask):
    num_units = 64
    act_opt_lr = 1.
    num_steps = 8
    lat_dep = 'ReLU'

    def get_dataset_params(self):
        p = ImageDatasetFactory.get_params()
        p.dataset = SupportedDatasets.MNIST
        p.datafolder = '/home/mshah1/workhorse3/'
        p.max_num_train = 50000
        p.max_num_test = 10000
        p.custom_transforms = (
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(32),
                torchvision.transforms.ToTensor()
            ]),
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(32),
                torchvision.transforms.ToTensor()
            ])
        )
        return p

    def get_model_params(self):
        fp: SequentialLayers.ModelParams = SequentialLayers.get_params()
        fp.common_params.input_size = [1, 32, 32]
        activate_logits = True

        p1: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        set_scanning_consistent_activation_layer_params(p1, self.num_units, True, self.lat_dep, self.act_opt_lr, 
                                                        self.num_steps, 5, 0, 3, 5, 5, nn.ReLU, 0.2,
                                                        activate_logits=activate_logits)

        p2: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        set_scanning_consistent_activation_layer_params(p2, self.num_units, True, self.lat_dep, self.act_opt_lr, 
                                                        self.num_steps, 3, 1, 2, 5, 5, nn.ReLU, 0.2,
                                                        activate_logits=activate_logits)
        
        fp.layer_params = [p1, p2]

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

    def get_experiment_params(self):
        nepochs = 30
        p = BaseExperimentConfig(
            ConsistentActivationModelAdversarialTrainer.TrainerParams(ConsistentActivationModelAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=20, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=True),
                AdversarialParams(testing_attack_params=get_apgd_inf_params([0.0, 0.025, 0.05, 0.1, 0.2, 0.3], 50)),
            ),
            SGDOptimizerConfig(lr=0.05, weight_decay=5e-5, momentum=0.9, nesterov=True),
            ReduceLROnPlateauConfig(),
            # OneCycleLRConfig(max_lr=1., epochs=nepochs, steps_per_epoch=176),
            logdir=LOGDIR, batch_size=256
        )
        return p

class MNISTConv2LConsistentActivation1StepsTask(MNISTConv2LConsistentActivation8StepsTask):
    num_steps = 1

class MNISTConv2LConsistentActivation0StepsTask(MNISTConv2LConsistentActivation8StepsTask):
    def get_model_params(self):
        p1: ConvEncoder.ModelParams = ConvEncoder.get_params()
        p1.common_params.input_size = [1, 32, 32]
        p1.common_params.activation = nn.ReLU
        p1.common_params.num_units = [64, 64]
        p1.common_params.dropout_p = 0.2
        p1.conv_params.kernel_sizes = [5, 3]
        p1.conv_params.padding = [0, 1]
        p1.conv_params.strides = [3, 2]

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
