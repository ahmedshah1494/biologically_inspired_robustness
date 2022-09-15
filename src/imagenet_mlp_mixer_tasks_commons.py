from copy import deepcopy
from re import L
from time import time
from typing import List, Tuple, Type
import numpy as np

import torchvision
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import (
    ConsistentActivationMixerBlock, ConsistentActivationMixerMLP,
    FirstNExtractionClassifier, LinearLayer, MixerBlock, MixerMLP, MLPMixer,
    NormalizationLayer, UnfoldPatchExtractor)
from adversarialML.biologically_inspired_models.src.models import (
    ConsistentActivationLayer, ConvEncoder, GeneralClassifier, IdentityLayer,
    ScanningConsistentActivationLayer, SequentialLayers, XResNet34, XResNet18, SupervisedContrastiveTrainingWrapper)
from adversarialML.biologically_inspired_models.src.retina_preproc import (
    AbstractRetinaFilter, RetinaBlurFilter, RetinaNonUniformPatchEmbedding,
    RetinaSampleFilter)
from adversarialML.biologically_inspired_models.src.supconloss import \
    TwoCropTransform
from adversarialML.biologically_inspired_models.src.trainers import (
    ActivityOptimizationSchedule, AdversarialParams, AdversarialTrainer,
    ConsistentActivationModelAdversarialTrainer,
    MixedPrecisionAdversarialTrainer)
from adversarialML.biologically_inspired_models.src.mlp_mixer_tasks import get_dataset_params
from mllib.adversarial.attacks import (AttackParamFactory, SupportedAttacks,
                                       SupportedBackend)
from mllib.datasets.dataset_factory import (ImageDatasetFactory,
                                            SupportedDatasets)
from mllib.models.base_models import MLP
from mllib.optimizers.configs import (AbstractOptimizerConfig, AbstractSchedulerConfig, AdamOptimizerConfig,
                                      CosineAnnealingWarmRestartsConfig,
                                      CyclicLRConfig, LinearLRConfig,
                                      ReduceLROnPlateauConfig,
                                      SequentialLRConfig, SGDOptimizerConfig)
from mllib.runners.configs import BaseExperimentConfig, TrainingParams
from mllib.tasks.base_tasks import AbstractTask
from torch import nn
from mllib.adversarial.attacks import TorchAttackAPGDInfParams

from mlp_mixer_tasks import get_resize_crop_flip_autoaugment_transforms

_LOGDIR = '/share/workhorse3/mshah1/biologically_inspired_models/logs/'
_EPS_LIST = [0.0, 0.008, 0.016, 0.024, 0.032, 0.048, 0.064]
_NEPOCHS = 300
_PATIENCE = 100
_APGD_STEPS = 50

def get_imagenet10_params(num_train=13_000, num_test=500, train_transforms=None, test_transforms=None):
    return get_dataset_params('/home/mshah1/workhorse3/imagenet-100/bin/64', SupportedDatasets.IMAGENET10, 
                                num_train, num_test, train_transforms, test_transforms)

def get_imagenet100_params(num_train=25, num_test=1, train_transforms=None, test_transforms=None):
    return get_dataset_params('/home/mshah1/workhorse3/imagenet-100/bin/64', SupportedDatasets.IMAGENET100, 
                                num_train, num_test, train_transforms, test_transforms)

def get_imagenet100_64_params(num_train=127500, num_test=1000, train_transforms=None, test_transforms=None):
    return get_dataset_params('/home/mshah1/workhorse3/imagenet-100/bin/64', SupportedDatasets.IMAGENET100_64, 
                                num_train, num_test, train_transforms, test_transforms)

def get_imagenet75_64_params(num_train=127500, num_test=1000, train_transforms=None, test_transforms=None):
    return get_dataset_params('/home/mshah1/workhorse3/imagenet-75/bin/64', SupportedDatasets.IMAGENET75_64, 
                                num_train, num_test, train_transforms, test_transforms)

def get_conv_patch_extractor_params(input_size, hidden_size, patch_size):
    patch_params: ConvEncoder.ModelParams = ConvEncoder.get_params()
    patch_params.common_params.input_size = input_size
    patch_params.common_params.num_units = [hidden_size]
    patch_params.common_params.activation = nn.Identity
    patch_params.conv_params.kernel_sizes = [patch_size]
    patch_params.conv_params.padding = [0]
    patch_params.conv_params.strides = [patch_size]
    npatches = (input_size[1] // patch_size)*(input_size[2] // patch_size)
    return patch_params, npatches

def get_retina_blur_conv_patch_extractor_params(input_size, hidden_size, patch_size, cone_std=0.12, rod_std=0.06, max_rod_density=0.12, kernel_size=16, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    rblur = RetinaBlurFilter.ModelParams(RetinaBlurFilter, input_size, cone_std=cone_std, rod_std=rod_std, max_rod_density=max_rod_density, kernel_size=kernel_size)
    norm = NormalizationLayer.ModelParams(NormalizationLayer, mean=mean, std=std)
    cnn_params, npatches = get_conv_patch_extractor_params(input_size, hidden_size, patch_size)

    p: SequentialLayers.ModelParams = SequentialLayers.get_params()
    p.common_params.input_size = input_size
    p.common_params.activation = nn.Identity
    p.layer_params = [rblur, norm, cnn_params]
    return p, npatches

def get_retina_nonuniform_conv_patch_extractor_params(input_size, hidden_size, loc_mode='random_uniform', mask_small_rf_region=False, isobox_w=None, rec_flds=None):
    if isobox_w is None:
        n_isoboxes = int(np.log2(min(input_size[1:]))) - 1
        isobox_w = [2**(i+1) for i in range(1,n_isoboxes)]

    if rec_flds is None:
        rec_flds = [2**i for i in range(len(isobox_w) + 1)]

    p = RetinaNonUniformPatchEmbedding.ModelParams(RetinaNonUniformPatchEmbedding, input_shape=input_size, hidden_size=hidden_size, loc_mode=loc_mode, 
                                                mask_small_rf_region=mask_small_rf_region, isobox_w=isobox_w, rec_flds=rec_flds)
    npatches = sum([(w//k)**2 for k,w in zip(rec_flds, isobox_w + [input_size[1]])])
    print(npatches, rec_flds, isobox_w)
    return p, npatches

def get_basic_mixer_mlp_params(activation, dropout_p, input_size, hidden_size):
    mlp_params: MixerMLP.ModelParams = MixerMLP.get_params()
    mlp_params.common_params.activation = activation
    mlp_params.common_params.dropout_p = dropout_p
    mlp_params.common_params.input_size = [input_size]
    mlp_params.common_params.num_units = hidden_size
    return mlp_params

def get_basic_mixer_block_params(mlpc_params, mlps_params, num_patches, hidden_size):
    block_params: MixerBlock.ModelParams = MixerBlock.get_params()
    block_params.channel_mlp_params = mlpc_params
    block_params.spatial_mlp_params = mlps_params
    block_params.common_params.input_size = [num_patches, hidden_size]
    return block_params

def get_linear_classifier_params(hidden_size, nclasses):
    cls_params: LinearLayer.ModelParams = LinearLayer.get_params()
    cls_params.common_params.input_size = hidden_size
    cls_params.common_params.num_units = nclasses
    cls_params.common_params.activation = nn.Identity
    return cls_params

def get_mlp_mixer_params(input_size, patch_params, cls_params, mixer_block_params, normalization_layer_params, normalize_input):
    mixer_params: MLPMixer.ModelParams = MLPMixer.get_params()
    mixer_params.common_params.input_size = input_size
    mixer_params.patch_gen_params = patch_params
    mixer_params.mixer_block_params = mixer_block_params
    mixer_params.classifier_params = cls_params
    mixer_params.normalize_input = normalize_input
    mixer_params.normalization_layer_params = normalization_layer_params
    return mixer_params

def get_basic_mlp_mixer_params(input_size, nclasses, patch_size, hidden_size, mlpc_hidden, mlps_hidden, activation, dropout_p, num_blocks, normalize_input=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    patch_params, num_patches = get_conv_patch_extractor_params(input_size, hidden_size, patch_size)
    mlpc_params = get_basic_mixer_mlp_params(activation, dropout_p, hidden_size, mlpc_hidden)
    mlps_params = get_basic_mixer_mlp_params(activation, dropout_p, num_patches, mlps_hidden)
    cls_params = get_linear_classifier_params(hidden_size, nclasses)
    mixer_block_params = get_basic_mixer_block_params(mlpc_params, mlps_params, num_patches, hidden_size)
    normalization_layer_params = NormalizationLayer.ModelParams(NormalizationLayer, mean=mean, std=std)
    mlp_mixer_params = get_mlp_mixer_params(input_size, patch_params, cls_params, [mixer_block_params]*num_blocks, normalization_layer_params, normalize_input)
    return mlp_mixer_params

def get_retina_blur_mlp_mixer_params(input_size, nclasses, patch_size, hidden_size, mlpc_hidden, mlps_hidden, 
                                        activation, dropout_p, num_blocks, 
                                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                                        cone_std=0.12, rod_std=0.06, max_rod_density=0.12, kernel_size=16):
    patch_params, num_patches = get_retina_blur_conv_patch_extractor_params(input_size, hidden_size, patch_size, cone_std, rod_std, max_rod_density, kernel_size, mean, std)
    mlpc_params = get_basic_mixer_mlp_params(activation, dropout_p, hidden_size, mlpc_hidden)
    mlps_params = get_basic_mixer_mlp_params(activation, dropout_p, num_patches, mlps_hidden)
    cls_params = get_linear_classifier_params(hidden_size, nclasses)
    mixer_block_params = get_basic_mixer_block_params(mlpc_params, mlps_params, num_patches, hidden_size)
    normalization_layer_params = NormalizationLayer.ModelParams(NormalizationLayer, mean=mean, std=std)
    mlp_mixer_params = get_mlp_mixer_params(input_size, patch_params, cls_params, [mixer_block_params]*num_blocks, normalization_layer_params, False)
    return mlp_mixer_params

def get_retina_nonuniform_patch_mlp_mixer_params(input_size, nclasses, hidden_size, mlpc_hidden, mlps_hidden, activation, dropout_p, num_blocks, loc_mode='random_uniform', mask_small_rf_region=False, isobox_w=None, rec_flds=None, normalize_input=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    patch_params, num_patches = get_retina_nonuniform_conv_patch_extractor_params(input_size, hidden_size, loc_mode, mask_small_rf_region, isobox_w, rec_flds)
    mlpc_params = get_basic_mixer_mlp_params(activation, dropout_p, hidden_size, mlpc_hidden)
    mlps_params = get_basic_mixer_mlp_params(activation, dropout_p, num_patches, mlps_hidden)
    cls_params = get_linear_classifier_params(hidden_size, nclasses)
    mixer_block_params = get_basic_mixer_block_params(mlpc_params, mlps_params, num_patches, hidden_size)
    normalization_layer_params = NormalizationLayer.ModelParams(NormalizationLayer, mean=mean, std=std)
    mlp_mixer_params = get_mlp_mixer_params(input_size, patch_params, cls_params, [mixer_block_params]*num_blocks, normalization_layer_params, normalize_input)
    return mlp_mixer_params

def get_adv_experiment_params(trainer_cls: Type[AdversarialTrainer], training_params: TrainingParams, adv_params:AdversarialParams,
                                optimizer_config:AbstractOptimizerConfig, scheduler_config: AbstractSchedulerConfig, batch_size: int,
                                exp_name: str = '', num_training=5):
    if isinstance(scheduler_config, CyclicLRConfig):
        training_params.scheduler_step_after_epoch = False
    p = BaseExperimentConfig(
        trainer_params=trainer_cls.TrainerParams(
            trainer_cls,
            training_params=training_params,
            adversarial_params=adv_params
        ),
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        batch_size=batch_size,
        logdir=_LOGDIR,
        exp_name=exp_name,
        num_trainings=num_training
    )
    return p

def get_apgd_inf_params(eps_list, nsteps, eot_iters=1):
    return [TorchAttackAPGDInfParams(eps=eps, nsteps=nsteps, eot_iter=eot_iters, seed=time()) for eps in eps_list]

def get_common_training_params():
    return TrainingParams(
        logdir=_LOGDIR, nepochs=_NEPOCHS, early_stop_patience=_PATIENCE, tracked_metric='val_accuracy', tracking_mode='max'
    )

def get_apgd_testing_adversarial_params():
    return AdversarialParams(
        testing_attack_params=get_apgd_inf_params(_EPS_LIST, _APGD_STEPS)
    )

def get_apgd_eot_testing_adversarial_params(n):
    return AdversarialParams(
        testing_attack_params=get_apgd_inf_params(_EPS_LIST, _APGD_STEPS, eot_iters=n)
    )