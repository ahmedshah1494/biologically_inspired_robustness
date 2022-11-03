import numpy as np
import torchvision
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import NormalizationLayer
from adversarialML.biologically_inspired_models.src.models import (ConsistentActivationClassifier, 
    SequentialLayers, ScanningConsistentActivationLayer, GeneralClassifier, ConvEncoder, LinearLayer,
    FlattenLayer, CommonModelParams, NormalizationLayer, ActivationLayer, DropoutLayer)
from adversarialML.biologically_inspired_models.src.retina_preproc import (
    GreyscaleLayer)
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

from ICASSP23.audio_feature_layer import MFCCExtractionLayer, ResamplingLayer

class MNISTAdvTrainMLP2048U4L02DropoutClassifier(AbstractTask):
    num_layers = 1
    num_units = 2048
    input_size = [1,28,28]
    dropout = 0.2
    eps = 0.3

    def get_dataset_params(self):
        p = get_mnist_params()
        return p
    
    def get_model_params(self):
        preproc_params = [NormalizationLayer.ModelParams(NormalizationLayer, [0.5], [0.5]), FlattenLayer.get_params()]
        linear_layer_params = [
            LinearLayer.ModelParams(LinearLayer, CommonModelParams(np.prod(self.input_size), self.num_units)),
            ActivationLayer.ModelParams(ActivationLayer, nn.ReLU),
            DropoutLayer.ModelParams(DropoutLayer, self.dropout),
            LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.num_units, 384)),
            ActivationLayer.ModelParams(ActivationLayer, nn.ReLU),
            DropoutLayer.ModelParams(DropoutLayer, self.dropout),
            LinearLayer.ModelParams(LinearLayer, CommonModelParams(384, self.num_units)),
            ActivationLayer.ModelParams(ActivationLayer, nn.ReLU),
            DropoutLayer.ModelParams(DropoutLayer, self.dropout),
            LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.num_units, self.num_units)),
            ActivationLayer.ModelParams(ActivationLayer, nn.ReLU),
            DropoutLayer.ModelParams(DropoutLayer, self.dropout),
        ]
        mlp_p = SequentialLayers.ModelParams(SequentialLayers, preproc_params+linear_layer_params)
        mlp_p.common_params.input_size = self.input_size

        cls_p = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.num_units, 10))

        p: GeneralClassifier.ModelParams = GeneralClassifier.get_params()
        p.feature_model_params = mlp_p
        p.classifier_params = cls_p
        return p
    
    def get_experiment_params(self):
        nepochs = 100
        p = BaseExperimentConfig(
            ConsistentActivationModelAdversarialTrainer.TrainerParams(ConsistentActivationModelAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=20, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=True),
                AdversarialParams(training_attack_params=get_pgd_inf_params([self.eps], 7, self.eps/4)[0],
                    testing_attack_params=get_apgd_inf_params([0.0, 0.025, 0.05, 0.1, 0.125, 0.15, 0.2], 50))
            ),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-5),
            ReduceLROnPlateauConfig(),
            logdir=LOGDIR, batch_size=256
        )
        return p

class FMNISTAdvTrainMLP2048U4L02DropoutClassifier(MNISTAdvTrainMLP2048U4L02DropoutClassifier):
    def get_dataset_params(self):
        p = get_fmnist_params()
        return p

class FMNISTAdvTrainMLP64U4L02DropoutClassifier(FMNISTAdvTrainMLP2048U4L02DropoutClassifier):
    num_units = 64
    eps = 0.1

class SpeechCommandsAdvTrainMLP2048U4L02DropoutClassifier(AbstractTask):
    input_size = [1,1,16000]
    num_units = 2048
    dropout = 0.2
    num_layers = 4
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
            FlattenLayer.get_params()
        ]

        linear_layer_params = [
            LinearLayer.ModelParams(LinearLayer, CommonModelParams(np.prod(self.input_size), self.num_units)),
            ActivationLayer.ModelParams(ActivationLayer, nn.ReLU),
            DropoutLayer.ModelParams(DropoutLayer, self.dropout),
            LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.num_units, self.num_units)),
            ActivationLayer.ModelParams(ActivationLayer, nn.ReLU),
            DropoutLayer.ModelParams(DropoutLayer, self.dropout),
            LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.num_units, self.num_units)),
            ActivationLayer.ModelParams(ActivationLayer, nn.ReLU),
            DropoutLayer.ModelParams(DropoutLayer, self.dropout),
            LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.num_units, self.num_units)),
            ActivationLayer.ModelParams(ActivationLayer, nn.ReLU),
            DropoutLayer.ModelParams(DropoutLayer, self.dropout),
        ]
        mlp_p = SequentialLayers.ModelParams(SequentialLayers, preproc_params+linear_layer_params)
        mlp_p.common_params.input_size = self.input_size

        cls_p = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.num_units, 10))

        p: GeneralClassifier.ModelParams = GeneralClassifier.get_params()
        p.feature_model_params = mlp_p
        p.classifier_params = cls_p
        return p

    def get_experiment_params(self):
        nepochs = 100
        p = BaseExperimentConfig(
            ConsistentActivationModelAdversarialTrainer.TrainerParams(ConsistentActivationModelAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=20, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=True),
                AdversarialParams(training_attack_params=get_pgd_inf_params([self.eps], 7, self.eps/4)[0],
                    testing_attack_params=get_apgd_inf_params([0.0, 0.00005, 0.0001, 0.00025, 0.0005, 0.001], 50))
            ),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-5),
            ReduceLROnPlateauConfig(),
            logdir=LOGDIR, batch_size=256
        )
        return p