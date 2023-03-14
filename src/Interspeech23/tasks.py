import os
import numpy as np
import torch
from adversarialML.biologically_inspired_models.src.models import (ConsistentActivationClassifier, 
    SequentialLayers, ScanningConsistentActivationLayer, GeneralClassifier, ConvEncoder, LinearLayer,
    FlattenLayer, CommonModelParams, NormalizationLayer, ActivationLayer, DropoutLayer, IdentityLayer)
from adversarialML.biologically_inspired_models.src.trainers import ConsistentActivationModelAdversarialTrainer, ActivityOptimizationParams, ActivityOptimizationSchedule
from mllib.optimizers.configs import (AdamOptimizerConfig,
                                      CosineAnnealingWarmRestartsConfig,
                                      CyclicLRConfig, LinearLRConfig,
                                      ReduceLROnPlateauConfig,
                                      SequentialLRConfig, OneCycleLRConfig, SGDOptimizerConfig)
from mllib.runners.configs import BaseExperimentConfig, TrainingParams
from mllib.tasks.base_tasks import AbstractTask
from torch import nn
from adversarialML.biologically_inspired_models.src.task_utils import *
from adversarialML.biologically_inspired_models.src.ICASSP23.audio_feature_layer import MFCCExtractionLayer, ResamplingLayer, LogMelSpectrogramExtractionLayer, MelSpectrogramExtractionLayer, SpecAugmentLayer
from adversarialML.biologically_inspired_models.src.Interspeech23.models import TDNNCTCASRModel, Conv1dEncoder, ConvParams, ScanningConsistentActivationLayer1d
from adversarialML.biologically_inspired_models.src.Interspeech23.trainers import SpeechAdversarialTrainer, ConsistentActivationSpeechAdversarialTrainer

NGPUS = len(os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(','))

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
    return p

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
    return p

class LibrispeechTDNN3L512UAsrCtc(AbstractTask):
    sample_rate = 16_000
    input_size = [1,sample_rate]
    num_units = 512
    sp_file = '/home/mshah1/workhorse3/librispeech/ls_train-char.model'
    def get_dataset_params(self):
        p = ImageDatasetFactory.get_params()
        p.dataset = SupportedDatasets.LIBRISPEECH
        p.datafolder = '/home/mshah1/workhorse3/librispeech'
        p.max_num_train = 281242
        p.max_num_test = 4000
        p.kwargs = {'sp_file': self.sp_file}
        return p
    
    def get_model_params(self):
        # preprocp = MFCCExtractionLayer.ModelParams(MFCCExtractionLayer, sample_rate=self.sample_rate, n_mfcc=40, log_mels=False, 
        #             melkwargs={
        #                 'n_fft':512, 'win_length': 32*(self.sample_rate//1000), 'hop_length':16*(self.sample_rate//1000)
        #             })
        melspecp = LogMelSpectrogramExtractionLayer.ModelParams(LogMelSpectrogramExtractionLayer, n_fft=512, win_length=16*(self.sample_rate//1000), 
                                                                hop_length=8*(self.sample_rate//1000), n_mels=80)
        specaugp = SpecAugmentLayer.ModelParams(SpecAugmentLayer, 27, 100)
        preprocp = SequentialLayers.ModelParams(SequentialLayers, [melspecp, specaugp], CommonModelParams(self.input_size))
        encp = Conv1dEncoder.ModelParams(Conv1dEncoder, CommonModelParams((80,self.sample_rate)), 
                                            [ConvParams(self.num_units,5,4), ConvParams(self.num_units,1,1), ConvParams(self.num_units,2,1)],
                                            group_norm=True)
        clsp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.num_units, 31))
        p = TDNNCTCASRModel.ModelParams(TDNNCTCASRModel, encp, clsp, preprocp, self.sp_file)
        return p
    
    def get_experiment_params(self):
        nepochs = 50
        p = BaseExperimentConfig(
            SpeechAdversarialTrainer.TrainerParams(SpeechAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=20, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False),
            ),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-5),
            # ReduceLROnPlateauConfig(),
            OneCycleLRConfig(max_lr=5e-3, epochs=nepochs, steps_per_epoch=4395, pct_start=0.0032, anneal_strategy='cos', div_factor=5e4, cycle_momentum=False),
            logdir=LOGDIR, batch_size=64
        )
        return p

class LibrispeechTDNN4L768UAsrCtc(AbstractTask):
    sample_rate = 16_000
    input_size = [1,sample_rate]
    num_units = 768
    sp_file = '/home/mshah1/workhorse3/librispeech/ls_train-char.model'
    def get_dataset_params(self):
        p = ImageDatasetFactory.get_params()
        p.dataset = SupportedDatasets.LIBRISPEECH
        p.datafolder = '/home/mshah1/workhorse3/librispeech'
        p.max_num_train = 281242
        p.max_num_test = 4000
        p.kwargs = {'sp_file': self.sp_file}
        return p
    
    def get_model_params(self):
        melspecp = LogMelSpectrogramExtractionLayer.ModelParams(LogMelSpectrogramExtractionLayer, n_fft=512, win_length=25*(self.sample_rate//1000), 
                                                                hop_length=10*(self.sample_rate//1000), n_mels=40)
        specaugp = SpecAugmentLayer.ModelParams(SpecAugmentLayer, 27, 100)
        preprocp = SequentialLayers.ModelParams(SequentialLayers, [melspecp], CommonModelParams(self.input_size))
        encp = Conv1dEncoder.ModelParams(Conv1dEncoder, CommonModelParams((40,self.sample_rate), dropout_p=0.25), 
                                            [ConvParams(self.num_units,13,2,6), ConvParams(self.num_units,17,2,8), ConvParams(self.num_units,21,1,10), ConvParams(self.num_units,25,1,12)],
                                            group_norm=True)
        clsp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.num_units, 31))
        p = TDNNCTCASRModel.ModelParams(TDNNCTCASRModel, encp, clsp, preprocp, self.sp_file)
        return p
    
    def get_experiment_params(self):
        nepochs = 20
        p = BaseExperimentConfig(
            SpeechAdversarialTrainer.TrainerParams(SpeechAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=20, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False),
            ),
            SGDOptimizerConfig(lr=0.05, weight_decay=5e-5, momentum=0.9),
            ReduceLROnPlateauConfig(patience=0, threshold=0.0025, factor=0.8, threshold_mode='abs'),
            # OneCycleLRConfig(max_lr=0.2, epochs=nepochs, steps_per_epoch=4395, pct_start=0.15, anneal_strategy='linear', div_factor=200, final_div_factor=1e6, cycle_momentum=False, three_phase=True),
            logdir=LOGDIR, batch_size=32
        )
        return p

class LibrispeechTDNN8L768UAsrCtc(LibrispeechTDNN4L768UAsrCtc):    
    def get_model_params(self):
        melspecp = LogMelSpectrogramExtractionLayer.ModelParams(LogMelSpectrogramExtractionLayer, n_fft=512, win_length=25*(self.sample_rate//1000), 
                                                                hop_length=10*(self.sample_rate//1000), n_mels=40)
        specaugp = SpecAugmentLayer.ModelParams(SpecAugmentLayer, 27, 100)
        preprocp = SequentialLayers.ModelParams(SequentialLayers, [melspecp], CommonModelParams(self.input_size))
        encp = Conv1dEncoder.ModelParams(Conv1dEncoder, CommonModelParams((40,self.sample_rate), dropout_p=0.25), 
                                            [
                                                ConvParams(self.num_units,13,2,6), ConvParams(self.num_units,15,2,7), ConvParams(self.num_units,17,1,8), ConvParams(self.num_units,19,1,9),
                                                ConvParams(self.num_units,21,1,10), ConvParams(self.num_units,23,1,11), ConvParams(self.num_units,25,1,12), ConvParams(self.num_units,27,1,13),
                                            ],
                                            group_norm=True)
        clsp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.num_units, 31))
        p = TDNNCTCASRModel.ModelParams(TDNNCTCASRModel, encp, clsp, preprocp, self.sp_file)
        return p
    
    def get_experiment_params(self):
        nepochs = 30
        per_gpu_batch_size = 32
        steps_per_epoch = int(np.ceil(self.get_dataset_params().max_num_train / (per_gpu_batch_size * NGPUS)))
        p = BaseExperimentConfig(
            SpeechAdversarialTrainer.TrainerParams(SpeechAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=20, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False),
            ),
            AdamOptimizerConfig(lr=0.05, weight_decay=5e-5),
            # ReduceLROnPlateauConfig(patience=0, threshold=0.0025, factor=0.8, threshold_mode='abs'),
            OneCycleLRConfig(max_lr=0.01, epochs=nepochs, steps_per_epoch=steps_per_epoch, pct_start=0.15, anneal_strategy='linear', div_factor=100, final_div_factor=1e6, cycle_momentum=False, three_phase=True),
            logdir=LOGDIR, batch_size=32
        )
        return p

class LibrispeechConsistentActivationTDNN1L512U16STask(AbstractTask):
    num_units = 512
    step_size = 0.25
    num_steps = 16
    lat_dep = 'ReLU'
    sample_rate = 16_000
    input_size = [1,sample_rate]
    sp_file = '/home/mshah1/workhorse3/librispeech/ls_train-char.model'

    def get_dataset_params(self):
        p = ImageDatasetFactory.get_params()
        p.dataset = SupportedDatasets.LIBRISPEECH
        p.datafolder = '/home/mshah1/workhorse3/librispeech'
        p.max_num_train = 281242
        p.max_num_test = 4000
        p.kwargs = {'sp_file': self.sp_file}
        return p

    def get_model_params(self):
        # preprocp = MFCCExtractionLayer.ModelParams(MFCCExtractionLayer, sample_rate=self.sample_rate, n_mfcc=40, log_mels=False, 
        #             melkwargs={
        #                 'n_fft':512, 'win_length': 32*(self.sample_rate//1000), 'hop_length':16*(self.sample_rate//1000)
        #             })
        melspecp = LogMelSpectrogramExtractionLayer.ModelParams(LogMelSpectrogramExtractionLayer, n_fft=512, win_length=16*(self.sample_rate//1000), 
                                                                hop_length=8*(self.sample_rate//1000), n_mels=80)
        # specaugp = SpecAugmentLayer.ModelParams(SpecAugmentLayer, 27, 100)
        # preprocp = SequentialLayers.ModelParams(SequentialLayers, [melspecp, specaugp], CommonModelParams(self.input_size))
        preprocp = melspecp
        ca_layer_params = ScanningConsistentActivationLayer1d.get_params()
        set_consistency_opt_params(ca_layer_params, True, 'ReLU', self.step_size, self.num_steps, self.num_steps, activate_logits=True)
        ca_layer_params.common_params = CommonModelParams([80,self.sample_rate])
        ca_layer_params.conv_params = ConvParams(self.num_units, 5, 4)
        clsp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.num_units, 31))
        p = TDNNCTCASRModel.ModelParams(TDNNCTCASRModel, ca_layer_params, clsp, preprocp, self.sp_file)
        return p

    def get_experiment_params(self):
        nepochs = 50
        p = BaseExperimentConfig(
            ConsistentActivationSpeechAdversarialTrainer.TrainerParams(ConsistentActivationSpeechAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=20, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False),
                act_opt_params=ActivityOptimizationParams(ActivityOptimizationSchedule.LINEAR, 1e-3)
            ),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-5),
            # ReduceLROnPlateauConfig(),
            OneCycleLRConfig(max_lr=.001, epochs=nepochs, steps_per_epoch=4395*2, pct_start=0.2, anneal_strategy='linear', cycle_momentum=False),
            logdir=LOGDIR, batch_size=32
        )
        return p

class LibrispeechConsistentActivationTDNN1L512U8STask(LibrispeechConsistentActivationTDNN1L512U16STask):
    num_steps = 8

class LibrispeechConsistentActivationTDNN2L512U8STask(LibrispeechConsistentActivationTDNN1L512U16STask):
    num_steps=8
    def get_model_params(self):
        melspecp = LogMelSpectrogramExtractionLayer.ModelParams(LogMelSpectrogramExtractionLayer, n_fft=512, win_length=16*(self.sample_rate//1000), 
                                                                hop_length=8*(self.sample_rate//1000), n_mels=80)
        # specaugp = SpecAugmentLayer.ModelParams(SpecAugmentLayer, 27, 100)
        # preprocp = SequentialLayers.ModelParams(SequentialLayers, [melspecp, specaugp], CommonModelParams(self.input_size))
        preprocp = melspecp

        p1 = ScanningConsistentActivationLayer1d.get_params()
        set_consistency_opt_params(p1, True, 'ReLU', self.step_size, self.num_steps, self.num_steps, activate_logits=True)
        p1.common_params = CommonModelParams([80,self.sample_rate])
        p1.conv_params = ConvParams(self.num_units, 5, 4)

        p2 = ScanningConsistentActivationLayer1d.get_params()
        set_consistency_opt_params(p2, True, 'ReLU', self.step_size, self.num_steps, self.num_steps, activate_logits=True)
        p2.common_params = CommonModelParams([self.num_units,self.sample_rate])
        p2.conv_params = ConvParams(self.num_units, 2, 1)

        ca_layer_params = SequentialLayers.ModelParams(SequentialLayers, [p1,p2], CommonModelParams([80,self.sample_rate]))

        clsp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.num_units, 31))
        p = TDNNCTCASRModel.ModelParams(TDNNCTCASRModel, ca_layer_params, clsp, preprocp, self.sp_file)
        return p
    
    def get_experiment_params(self):
        nepochs = 50
        p = BaseExperimentConfig(
            ConsistentActivationSpeechAdversarialTrainer.TrainerParams(ConsistentActivationSpeechAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=20, tracked_metric='val_accuracy',
                    tracking_mode='min', scheduler_step_after_epoch=False),
                act_opt_params=ActivityOptimizationParams(ActivityOptimizationSchedule.LINEAR, 1e-3)
            ),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-5),
            # ReduceLROnPlateauConfig(),
            OneCycleLRConfig(max_lr=.001, epochs=nepochs, steps_per_epoch=4395*2, pct_start=0.2, anneal_strategy='linear', cycle_momentum=False),
            logdir=LOGDIR, batch_size=32
        )
        return p

class LibrispeechConsistentActivationTDNN2L512U8S025DropoutTask(LibrispeechConsistentActivationTDNN1L512U16STask):
    num_steps = 8
    num_mels = 40
    num_units = 768
    dropout = 0.25
    def get_model_params(self):
        melspecp = LogMelSpectrogramExtractionLayer.ModelParams(LogMelSpectrogramExtractionLayer, n_fft=512, win_length=25*(self.sample_rate//1000), 
                                                                hop_length=10*(self.sample_rate//1000), n_mels=self.num_mels)
        # specaugp = SpecAugmentLayer.ModelParams(SpecAugmentLayer, 27, 100)
        # preprocp = SequentialLayers.ModelParams(SequentialLayers, [melspecp, specaugp], CommonModelParams(self.input_size))
        preprocp = melspecp

        ca_layer_params = []
        kernsz = [17,8]
        stride = [4,1]
        padding = [8,11]
        dilation = [1,3]
        for k,s,p,d in zip(kernsz, stride, padding, dilation):
            cap = ScanningConsistentActivationLayer1d.get_params()
            set_consistency_opt_params(cap, True, 'ReLU', self.step_size, self.num_steps, self.num_steps, activate_logits=True)
            cap.common_params = CommonModelParams([self.num_mels,self.sample_rate])
            cap.common_params.dropout_p = self.dropout
            cap.conv_params = ConvParams(self.num_units, k, s, p, d)
            ca_layer_params.append(cap)

        ca_layer_params = SequentialLayers.ModelParams(SequentialLayers, ca_layer_params, CommonModelParams([self.num_mels,self.sample_rate]))

        clsp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.num_units, 31))
        p = TDNNCTCASRModel.ModelParams(TDNNCTCASRModel, ca_layer_params, clsp, preprocp, self.sp_file)
        return p
    
    def get_experiment_params(self):
        nepochs = 50
        p = BaseExperimentConfig(
            ConsistentActivationSpeechAdversarialTrainer.TrainerParams(ConsistentActivationSpeechAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=20, tracked_metric='val_accuracy',
                    tracking_mode='min', scheduler_step_after_epoch=False),
                # act_opt_params=ActivityOptimizationParams(ActivityOptimizationSchedule.LINEAR, 1e-3)
            ),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-5),
            # ReduceLROnPlateauConfig(),
            OneCycleLRConfig(max_lr=.001, epochs=nepochs, steps_per_epoch=4395*4, pct_start=0.2, anneal_strategy='linear', cycle_momentum=False),
            logdir=LOGDIR, batch_size=16
        )
        return p

class LibrispeechConsistentActivationTDNN4L512U8S025DropoutTask(LibrispeechConsistentActivationTDNN1L512U16STask):
    num_steps = 8
    num_mels = 40
    num_units = 768
    dropout = 0.25
    def get_model_params(self):
        melspecp = LogMelSpectrogramExtractionLayer.ModelParams(LogMelSpectrogramExtractionLayer, n_fft=512, win_length=25*(self.sample_rate//1000), 
                                                                hop_length=10*(self.sample_rate//1000), n_mels=self.num_mels)
        # specaugp = SpecAugmentLayer.ModelParams(SpecAugmentLayer, 27, 100)
        # preprocp = SequentialLayers.ModelParams(SequentialLayers, [melspecp, specaugp], CommonModelParams(self.input_size))
        preprocp = melspecp

        ca_layer_params = []
        kernsz = [13,7,11,12]
        stride = [4,1,1,1]
        padding = [6,8,10,12]
        dilation = [1,3,2,2]
        for k,s,p,d in zip(kernsz, stride, padding, dilation):
            cap = ScanningConsistentActivationLayer1d.get_params()
            set_consistency_opt_params(cap, True, 'ReLU', self.step_size, self.num_steps, self.num_steps, activate_logits=True)
            cap.common_params = CommonModelParams([self.num_mels,self.sample_rate])
            cap.common_params.dropout_p = self.dropout
            cap.conv_params = ConvParams(self.num_units, k, s, p, d)
            ca_layer_params.append(cap)

        ca_layer_params = SequentialLayers.ModelParams(SequentialLayers, ca_layer_params, CommonModelParams([self.num_mels,self.sample_rate]))

        clsp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.num_units, 31))
        p = TDNNCTCASRModel.ModelParams(TDNNCTCASRModel, ca_layer_params, clsp, preprocp, self.sp_file)
        return p
    
    def get_experiment_params(self):
        nepochs = 50
        p = BaseExperimentConfig(
            ConsistentActivationSpeechAdversarialTrainer.TrainerParams(ConsistentActivationSpeechAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=20, tracked_metric='val_accuracy',
                    tracking_mode='min', scheduler_step_after_epoch=False),
                # act_opt_params=ActivityOptimizationParams(ActivityOptimizationSchedule.LINEAR, 1e-3)
            ),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-5),
            # ReduceLROnPlateauConfig(),
            OneCycleLRConfig(max_lr=.001, epochs=nepochs, steps_per_epoch=4395*8, pct_start=0.2, anneal_strategy='linear', cycle_momentum=False),
            logdir=LOGDIR, batch_size=8
        )
        return p