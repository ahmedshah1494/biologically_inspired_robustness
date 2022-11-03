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
from ICASSP23.ca_tasks import set_consistency_opt_params

class MNISTAdvTrainConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier(AbstractTask):
    num_steps = 16
    step_size = 0.25
    dropout = 0.2
    num_units = 2048
    input_size = [1,28,28]
    eps = 0.3

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
                AdversarialParams(
                    training_attack_params=get_pgd_inf_params([self.eps], 7, self.eps/4)[0],
                    testing_attack_params=get_apgd_inf_params([0.0, 0.025, 0.05, 0.1, 0.125, 0.15, 0.2], 50)
                )
            ),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-5),
            ReduceLROnPlateauConfig(),
            logdir=LOGDIR, batch_size=256
        )
        return p

class FMNISTAdvTrainConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier(MNISTAdvTrainConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier):
    def get_dataset_params(self):
        p = get_fmnist_params()
        return p

class SpeechCommandsAdvTrainConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier(AbstractTask):
    input_size = [1,1,16000]
    num_units = 2048
    step_size = 0.25
    num_steps = 16
    dropout = 0.2
    eps = 0.00025
    
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
                AdversarialParams(
                    training_attack_params=get_pgd_inf_params([self.eps], 7, self.eps/4)[0],
                    testing_attack_params=get_apgd_inf_params([0.0, 0.00005, 0.0001, 0.00025, 0.0005, 0.001], 50)
                )
            ),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-5),
            ReduceLROnPlateauConfig(),
            logdir=LOGDIR, batch_size=256
        )
        return p