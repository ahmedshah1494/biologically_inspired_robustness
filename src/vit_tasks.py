import torch
import torchvision
from adversarialML.biologically_inspired_models.src.vit_models import ViTClassifier
from adversarialML.biologically_inspired_models.src.trainers import MixedPrecisionAdversarialTrainer, LightningAdversarialTrainer
from mllib.optimizers.configs import (AdamOptimizerConfig,
                                      CosineAnnealingWarmRestartsConfig,
                                      CyclicLRConfig, LinearLRConfig,
                                      ReduceLROnPlateauConfig,
                                      SequentialLRConfig, OneCycleLRConfig, SGDOptimizerConfig)
from adversarialML.biologically_inspired_models.src.retina_preproc import (
    RetinaBlurFilter, RetinaNonUniformPatchEmbedding, RetinaWarp, GaussianNoiseLayer)
from adversarialML.biologically_inspired_models.src.models import (
    CommonModelParams, GeneralClassifier, SequentialLayers, XResNet34, XResNet18, XResNet50, WideResnet,
    ActivationLayer, BatchNorm2DLayer, LogitAverageEnsembler)
from mllib.runners.configs import BaseExperimentConfig, TrainingParams
from mllib.tasks.base_tasks import AbstractTask
from torch import nn
from adversarialML.biologically_inspired_models.src.task_utils import *

from task_utils import get_ecoset100folder_params, get_ecoset10_params, get_ecoset10folder_params

class Cifar10CyclicLRAutoAugmentViT(AbstractTask):
    def get_dataset_params(self):
        p = get_cifar10_params(num_train=50_000)
        p.custom_transforms = (
            torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                torchvision.transforms.Resize(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.CIFAR10),
                torchvision.transforms.ToTensor()
            ]),
            torchvision.transforms.ToTensor()
        )
        return p

    def get_model_params(self):
        return ViTClassifier.ModelParams(ViTClassifier, num_labels=10)

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 30
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-5, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=1., epochs=nepochs, steps_per_epoch=352),
            logdir=LOGDIR, batch_size=128
        )

class EcosetNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentViTB16(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
    noise_std = 0.125
    def get_dataset_params(self) :
        p = get_ecoset_params(train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.RandomCrop(self.imgs_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandAugment(magnitude=15)
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        return p

    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform')
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        vit_p = ViTClassifier.ModelParams(ViTClassifier, num_labels=565)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, vit_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        parallelism = torch.cuda.device_count() / 2
        nepochs = int(1 * torch.cuda.device_count() / parallelism)
        cfg =  BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            # SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            # OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=1839, pct_start=0.05, anneal_strategy='linear'),
            # OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=5632, pct_start=0.1, anneal_strategy='cos', div_factor=10, three_phase=True),
            AdamOptimizerConfig(weight_decay=0.1),
            OneCycleLRConfig(max_lr=0.001, epochs=nepochs, steps_per_epoch=int(parallelism * 11264 // torch.cuda.device_count()), pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=int(128 / parallelism)
        )
        print(cfg)
        return cfg

class EcosetNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentViTB16_2(EcosetNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentViTB16):
    pass

class EcosetNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentViTB16_3(EcosetNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentViTB16):
    pass

class EcosetNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentViTB16_4(EcosetNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentViTB16):
    pass

class EcosetNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentViTB16_6(EcosetNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentViTB16):
    pass

class EcosetNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentViTB16_8(EcosetNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentViTB16):
    pass

class EcosetNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentViTB16_16(EcosetNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentViTB16):
    pass