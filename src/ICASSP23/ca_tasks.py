import numpy as np
import torchvision
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import NormalizationLayer
from adversarialML.biologically_inspired_models.src.models import (ConsistentActivationClassifier, 
    SequentialLayers, ScanningConsistentActivationLayer, GeneralClassifier, ConvEncoder, LinearLayer,
    FlattenLayer, CommonModelParams, NormalizationLayer, ConsistentActivationLayer)
from adversarialML.biologically_inspired_models.src.retina_preproc import (
    GreyscaleLayer)
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
from ICASSP23.audio_feature_layer import MFCCExtractionLayer, ResamplingLayer

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

class MNISTConsistentActivation64U1L1SClassifier(AbstractTask):
    num_layers = 1
    num_steps = 1
    step_size = 1.
    cls_num_steps = 1
    cls_step_size = 1.
    dropout = 0.
    num_units = 64
    input_size = [1,28,28]

    def get_dataset_params(self):
        p = get_mnist_params()
        return p
    
    def get_model_params(self):
        preproc_params = [NormalizationLayer.ModelParams(NormalizationLayer, [0.5], [0.5]), FlattenLayer.get_params()]
        ca_layer_params = ConsistentActivationLayer.get_params()
        set_consistency_opt_params(ca_layer_params, True, 'ReLU', self.step_size, self.num_steps, self.num_steps, activate_logits=True)
        ca_layer_params.common_params.input_size = np.prod(self.input_size)
        ca_layer_params.common_params.num_units = self.num_units
        ca_layer_params.common_params.num_units = self.num_units
        ca_layer_params.common_params.dropout_p = self.dropout
        feat_params = SequentialLayers.ModelParams(SequentialLayers, preproc_params+[ca_layer_params])
        feat_params.common_params.input_size = self.input_size

        cls_p = LinearLayer.ModelParams(LinearLayer, CommonModelParams(np.prod(self.input_size), 10, nn.Identity))

        # cls_p: ConsistentActivationClassifier.ModelParams = ConsistentActivationClassifier.get_params()
        # set_consistency_opt_params(cls_p, True, 'Linear', self.cls_step_size, self.cls_num_steps, self.cls_num_steps, activate_logits=False)
        # cls_p.common_params.input_size = np.prod(self.input_size)
        # cls_p.common_params.activation = nn.Identity
        # cls_p.common_params.num_units = 10
        # cls_p.classification_params.num_classes = 10

        p: GeneralClassifier.ModelParams = GeneralClassifier.get_params()
        p.feature_model_params = feat_params
        p.classifier_params = cls_p
        return p
    
    def get_experiment_params(self):
        nepochs = 100
        p = BaseExperimentConfig(
            ConsistentActivationModelAdversarialTrainer.TrainerParams(ConsistentActivationModelAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=20, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=True),
                AdversarialParams(testing_attack_params=get_apgd_inf_params([0.0, 0.025, 0.05, 0.1, 0.125, 0.15, 0.2], 50))
            ),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-5),
            ReduceLROnPlateauConfig(),
            logdir=LOGDIR, batch_size=256
        )
        return p

class MNISTConsistentActivation128U1L1SClassifier(MNISTConsistentActivation64U1L1SClassifier):
    num_units = 128

class MNISTConsistentActivation256U1L1SClassifier(MNISTConsistentActivation64U1L1SClassifier):
    num_units = 256

class MNISTConsistentActivation512U1L1SClassifier(MNISTConsistentActivation64U1L1SClassifier):
    num_units = 512

class MNISTConsistentActivation1024U1L1SClassifier(MNISTConsistentActivation64U1L1SClassifier):
    num_units = 1024

class MNISTConsistentActivation2048U1L1SClassifier(MNISTConsistentActivation64U1L1SClassifier):
    num_units = 2048

class MNISTConsistentActivation3072U1L1SClassifier(MNISTConsistentActivation64U1L1SClassifier):
    num_units = 3072

class MNISTConsistentActivation4096U1L1SClassifier(MNISTConsistentActivation64U1L1SClassifier):
    num_units = 4096

class MNISTConsistentActivation2048U1L1S075ActOptLRClassifier(MNISTConsistentActivation2048U1L1SClassifier):
    step_size = 0.75

class MNISTConsistentActivation2048U1L1S05ActOptLRClassifier(MNISTConsistentActivation2048U1L1SClassifier):
    step_size = 0.5

class MNISTConsistentActivation2048U1L1S025ActOptLRClassifier(MNISTConsistentActivation2048U1L1SClassifier):
    step_size = 0.25

class MNISTConsistentActivation2048U1L1S0125ActOptLRClassifier(MNISTConsistentActivation2048U1L1SClassifier):
    step_size = 0.125

class MNISTConsistentActivation2048U1L1S0ActOptLRClassifier(MNISTConsistentActivation2048U1L1SClassifier):
    step_size = 0.

class MNISTConsistentActivation2048U1L1S0ClsSClassifier(MNISTConsistentActivation2048U1L1SClassifier):
    num_steps = 1
    step_size = 1.
    cls_num_steps = 0

class MNISTConsistentActivation2048U1L2S0ClsSClassifier(MNISTConsistentActivation2048U1L1SClassifier):
    num_steps = 2
    step_size = 1./2
    cls_num_steps = 0

class MNISTConsistentActivation2048U1L4S0ClsSClassifier(MNISTConsistentActivation2048U1L1SClassifier):
    num_steps = 4
    step_size = 1./4
    cls_num_steps = 0

class MNISTConsistentActivation2048U1L8S0ClsSClassifier(MNISTConsistentActivation2048U1L1SClassifier):
    num_steps = 8
    step_size = 1/8
    cls_num_steps = 0

class MNISTConsistentActivation2048U1L16S0ClsSClassifier(MNISTConsistentActivation2048U1L1SClassifier):
    num_steps = 16
    step_size = 1./8
    cls_num_steps = 0

class MNISTConsistentActivation2048U1L16S0ClsS075ActOptLRClassifier(MNISTConsistentActivation2048U1L1SClassifier):
    num_steps = 16
    step_size = 0.75
    cls_num_steps = 0

class MNISTConsistentActivation2048U1L16S0ClsS05ActOptLRClassifier(MNISTConsistentActivation2048U1L1SClassifier):
    num_steps = 16
    step_size = 0.5
    cls_num_steps = 0

class MNISTConsistentActivation2048U1L16S0ClsS025ActOptLRClassifier(MNISTConsistentActivation2048U1L1SClassifier):
    num_steps = 16
    step_size = 0.25
    cls_num_steps = 0

# class MNISTConsistentActivation2048U1L16S1ClsS025ActOptLRClassifier(MNISTConsistentActivation2048U1L1SClassifier):
#     num_steps = 16
#     step_size = 0.25
#     cls_num_steps = 1
#     cls_step_size = 0.25

# class MNISTConsistentActivation2048U1L16S2ClsS025ActOptLRClassifier(MNISTConsistentActivation2048U1L1SClassifier):
#     num_steps = 16
#     step_size = 0.25
#     cls_num_steps = 2
#     cls_step_size = 0.25

# class MNISTConsistentActivation2048U1L16S4ClsS025_0125ActOptLRClassifier(MNISTConsistentActivation2048U1L1SClassifier):
#     num_steps = 16
#     step_size = 0.25
#     cls_num_steps = 4
#     cls_step_size = 0.125

# class MNISTConsistentActivation2048U1L16S8ClsS025_00625ActOptLRClassifier(MNISTConsistentActivation2048U1L1SClassifier):
#     num_steps = 16
#     step_size = 0.25
#     cls_num_steps = 8
#     cls_step_size = 0.0625

# class MNISTConsistentActivation2048U1L16S16ClsS025ActOptLRClassifier(MNISTConsistentActivation2048U1L1SClassifier):
#     num_steps = 16
#     step_size = 0.25
#     cls_num_steps = 16
#     cls_step_size = 0.125

# class MNISTConsistentActivation2048U1L16S0ClsS00625ActOptLRClassifier(MNISTConsistentActivation2048U1L1SClassifier):
#     num_steps = 16
#     step_size = 0.0625
#     cls_num_steps = 0

# class MNISTConsistentActivation2048U1L16S0ClsS025ActOptLR01DropoutClassifier(MNISTConsistentActivation2048U1L16S0ClsS025ActOptLRClassifier):
#     dropout = 0.1

class MNISTConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier(MNISTConsistentActivation2048U1L16S0ClsS025ActOptLRClassifier):
    dropout = 0.2

# class MNISTConsistentActivation2048U1L16S0ClsS025ActOptLR03DropoutClassifier(MNISTConsistentActivation2048U1L16S0ClsS025ActOptLRClassifier):
#     dropout = 0.3

class MNISTConsistentActivation64U1L16S0ClsS025ActOptLR02DropoutClassifier(MNISTConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier):
    num_units = 64

class FMNISTConsistentActivation64U1L1SClassifier(MNISTConsistentActivation64U1L1SClassifier):
    num_layers = 1
    num_steps = 1
    step_size = 1.
    cls_num_steps = 1
    cls_step_size = 1.
    num_units = 64
    input_size = [1,28,28]

    def get_dataset_params(self):
        p = get_fmnist_params()
        return p

class FMNISTConsistentActivation1024U1L1SClassifier(FMNISTConsistentActivation64U1L1SClassifier):
    num_units = 1024

class FMNISTConsistentActivation2048U1L1SClassifier(FMNISTConsistentActivation64U1L1SClassifier):
    num_units = 2048

class FMNISTConsistentActivation3072U1L1SClassifier(FMNISTConsistentActivation64U1L1SClassifier):
    num_units = 3072

class FMNISTConsistentActivation4096U1L1SClassifier(FMNISTConsistentActivation64U1L1SClassifier):
    num_units = 4096

class FMNISTConsistentActivation2048U1L1S0ClsSClassifier(FMNISTConsistentActivation2048U1L1SClassifier):
    num_steps = 1
    step_size = 1.
    cls_num_steps = 0

class FMNISTConsistentActivation2048U1L2S0ClsSClassifier(FMNISTConsistentActivation2048U1L1SClassifier):
    num_steps = 2
    step_size = 1./2
    cls_num_steps = 0

class FMNISTConsistentActivation2048U1L4S0ClsSClassifier(FMNISTConsistentActivation2048U1L1SClassifier):
    num_steps = 4
    step_size = 1./4
    cls_num_steps = 0

class FMNISTConsistentActivation2048U1L8S0ClsSClassifier(FMNISTConsistentActivation2048U1L1SClassifier):
    num_steps = 8
    step_size = 1/4
    cls_num_steps = 0

class FMNISTConsistentActivation2048U1L16S0ClsSClassifier(FMNISTConsistentActivation2048U1L1SClassifier):
    num_steps = 16
    step_size = 1./8
    cls_num_steps = 0

class FMNISTConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier(FMNISTConsistentActivation64U1L1SClassifier):
    num_units = 2048
    step_size = 0.25
    num_steps = 16
    dropout = 0.2

class FMNISTConsistentActivation2048U1L8S0ClsS025ActOptLR02DropoutClassifier(FMNISTConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier):
    num_steps = 8

class FMNISTConsistentActivation2048U1L4S0ClsS025ActOptLR02DropoutClassifier(FMNISTConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier):
    num_steps = 4

class FMNISTConsistentActivation2048U1L2S0ClsS025ActOptLR02DropoutClassifier(FMNISTConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier):
    num_steps = 2

class FMNISTConsistentActivation2048U1L1S0ClsS025ActOptLR02DropoutClassifier(FMNISTConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier):
    num_steps = 1

class FMNISTConsistentActivation64U1L16S0ClsS025ActOptLR02DropoutClassifier(FMNISTConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier):
    num_units = 64

class SpeechCommandsConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier(AbstractTask):
    input_size = [1,1,16000]
    num_units = 2048
    step_size = 0.25
    num_steps = 16
    dropout = 0.2
    
    def get_dataset_params(self):
        p = ImageDatasetFactory.get_params()
        p.dataset = SupportedDatasets.SPEECHCOMMANDS
        p.datafolder = '/home/mshah1/workhorse3/'
        p.max_num_train = 40000
        p.max_num_test = 4000
        return p
    
    def get_model_params(self):
        preproc_params = [
            ResamplingLayer.ModelParams(ResamplingLayer, orig_freq=16_000, new_freq=8000),
            MFCCExtractionLayer.ModelParams(MFCCExtractionLayer, sample_rate=8000, n_mfcc=16, log_mels=True, 
                melkwargs={
                    'n_fft':512, 'win_length': 64, 'hop_length':32
                }),
            NormalizationLayer.ModelParams(NormalizationLayer, -0.9672, 9.2037),
            ]
        ca_layer_params = ConsistentActivationLayer.get_params()
        set_consistency_opt_params(ca_layer_params, True, 'ReLU', self.step_size, self.num_steps, self.num_steps, activate_logits=True)
        ca_layer_params.common_params.input_size = np.prod(self.input_size)
        ca_layer_params.common_params.num_units = self.num_units
        ca_layer_params.common_params.num_units = self.num_units
        ca_layer_params.common_params.dropout_p = self.dropout
        feat_params = SequentialLayers.ModelParams(SequentialLayers, preproc_params+[ca_layer_params])
        feat_params.common_params.input_size = self.input_size

        cls_p = LinearLayer.ModelParams(LinearLayer, CommonModelParams(np.prod(self.input_size), 10, nn.Identity))
        p: GeneralClassifier.ModelParams = GeneralClassifier.get_params()
        p.feature_model_params = feat_params
        p.classifier_params = cls_p
        return p

    def get_experiment_params(self):
        nepochs = 100
        p = BaseExperimentConfig(
            ConsistentActivationModelAdversarialTrainer.TrainerParams(ConsistentActivationModelAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=20, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=True),
                AdversarialParams(testing_attack_params=get_apgd_inf_params([0.0, 0.00005, 0.0001, 0.00025, 0.0005, 0.001], 50))
            ),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-5),
            ReduceLROnPlateauConfig(),
            logdir=LOGDIR, batch_size=256
        )
        return p

class SpeechCommandsConsistentActivation2048U1L8S0ClsS025ActOptLR02DropoutClassifier(SpeechCommandsConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier):
    num_steps = 8

class SpeechCommandsConsistentActivation2048U1L4S0ClsS025ActOptLR02DropoutClassifier(SpeechCommandsConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier):
    num_steps = 4

class SpeechCommandsConsistentActivation2048U1L2S0ClsS025ActOptLR02DropoutClassifier(SpeechCommandsConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier):
    num_steps = 2

class SpeechCommandsConsistentActivation2048U1L1S0ClsS025ActOptLR02DropoutClassifier(SpeechCommandsConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier):
    num_steps = 1