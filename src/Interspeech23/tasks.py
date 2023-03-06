import numpy as np
from adversarialML.biologically_inspired_models.src.models import (ConsistentActivationClassifier, 
    SequentialLayers, ScanningConsistentActivationLayer, GeneralClassifier, ConvEncoder, LinearLayer,
    FlattenLayer, CommonModelParams, NormalizationLayer, ActivationLayer, DropoutLayer, IdentityLayer)
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
from adversarialML.biologically_inspired_models.src.ICASSP23.audio_feature_layer import MFCCExtractionLayer, ResamplingLayer
from adversarialML.biologically_inspired_models.src.Interspeech23.models import TDNNCTCASRModel, Conv1dEncoder, ConvParams
from adversarialML.biologically_inspired_models.src.Interspeech23.trainers import SpeechAdversarialTrainer

class TDNN1L64UAsrCtc(AbstractTask):
    sample_rate = 16_000
    input_size = [1,sample_rate]
    def get_dataset_params(self):
        p = ImageDatasetFactory.get_params()
        p.dataset = SupportedDatasets.LIBRISPEECH
        p.datafolder = '/home/mshah1/workhorse3/librispeech'
        p.max_num_train = 200000
        p.max_num_test = 4000
        return p
    
    def get_model_params(self):
        preprocp = MFCCExtractionLayer.ModelParams(MFCCExtractionLayer, sample_rate=self.sample_rate, n_mfcc=40, log_mels=False, 
                    melkwargs={
                        'n_fft':512, 'win_length': 32*(self.sample_rate//1000), 'hop_length':4*(self.sample_rate//1000)
                    })
        encp = Conv1dEncoder.ModelParams(Conv1dEncoder, CommonModelParams((40,16_000), [1024, 2048, 1024]), 
                                            ConvParams([5, 2, 2],[4, 2, 2],[0, 0, 0],[1, 1, 1]))
        clsp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(1024, 1000))
        p = TDNNCTCASRModel.ModelParams(TDNNCTCASRModel, encp, clsp, preprocp, '/home/mshah1/workhorse3/librispeech/ls_train-v1000.model')
        return p
    
    def get_experiment_params(self):
        nepochs = 50
        p = BaseExperimentConfig(
            SpeechAdversarialTrainer.TrainerParams(SpeechAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=20, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False),
                AdversarialParams(testing_attack_params=get_apgd_inf_params([0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1], 50))
            ),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-5),
            # ReduceLROnPlateauConfig(),
            OneCycleLRConfig(max_lr=.001, epochs=nepochs, steps_per_epoch=3125, pct_start=0.2),
            logdir=LOGDIR, batch_size=64
        )
        return p