import torchvision
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import NormalizationLayer
from adversarialML.biologically_inspired_models.src.models import (
    CommonModelParams, GeneralClassifier, SequentialLayers, XResNet34, XResNet18, XResNet50, WideResnet,
    ActivationLayer, BatchNorm2DLayer)
from adversarialML.biologically_inspired_models.src.retina_preproc import (
    RetinaBlurFilter, RetinaNonUniformPatchEmbedding, RetinaWarp, GaussianBlurLayer, GaussianNoiseLayer)
from adversarialML.biologically_inspired_models.src.supconloss import \
    TwoCropTransform
from adversarialML.biologically_inspired_models.src.trainers import MixedPrecisionAdversarialTrainer, LightningAdversarialTrainer
from mllib.optimizers.configs import (AdamOptimizerConfig,
                                      CosineAnnealingWarmRestartsConfig,
                                      CyclicLRConfig, LinearLRConfig,
                                      ReduceLROnPlateauConfig,
                                      SequentialLRConfig, OneCycleLRConfig, SGDOptimizerConfig)
from mllib.runners.configs import BaseExperimentConfig, TrainingParams
from mllib.tasks.base_tasks import AbstractTask
from torch import nn
from adversarialML.biologically_inspired_models.src.task_utils import *


class Ecoset10GaussianBlurCyclicLRRandAugmentXResNet18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 1
    def get_dataset_params(self) :
        p = get_ecoset10_params(train_transforms=[
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
        rblur_p = GaussianBlurLayer.ModelParams(GaussianBlurLayer, std=10.5)
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rblur_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=376, pct_start=0.2, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128
        )

class Ecoset10GaussianBlurCyclicLRRandAugmentXResNet2x18(Ecoset10GaussianBlurCyclicLRRandAugmentXResNet18):
    widen_factor = 2

class Ecoset10GaussianBlurCyclicLRRandAugmentXResNet4x18(Ecoset10GaussianBlurCyclicLRRandAugmentXResNet18):
    widen_factor = 4

class Ecoset10GaussianBlurCyclicLRRandAugmentXResNet8x18(Ecoset10GaussianBlurCyclicLRRandAugmentXResNet18):
    widen_factor = 8

class Ecoset10GaussianBlurCyclicLR1e_1RandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
    def get_dataset_params(self) :
        p = get_ecoset10_params(train_transforms=[
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
        rblur_p = GaussianBlurLayer.ModelParams(GaussianBlurLayer, std=10.5)
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rblur_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=375, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128
        )

class Ecoset10NoisyGaussianBlurS2500CyclicLR1e_1RandAugmentXResNet2x18(Ecoset10GaussianBlurCyclicLR1e_1RandAugmentXResNet2x18):
    noise_std = 0.25
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = GaussianBlurLayer.ModelParams(GaussianBlurLayer, std=10.5)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p

class Ecoset100GaussianBlurCyclicLRRandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
    def get_dataset_params(self) :
        p = get_ecoset100folder_params(num_train=500000, train_transforms=[
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
        rblur_p = GaussianBlurLayer.ModelParams(GaussianBlurLayer, std=10.5)
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 100), num_classes=100,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rblur_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 40
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=1839, pct_start=0.05, anneal_strategy='cos'),
            logdir=LOGDIR, batch_size=256
        )

class Cifar10GaussianBlurCyclicLRAutoAugmentWideResNet1x22(AbstractTask):
    input_size = [3, 32, 32]
    widen_factor = 1
    depth = 22
    def get_dataset_params(self):
        p = get_cifar10_params(num_train=50_000)
        p.custom_transforms = (
            torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.CIFAR10),
                torchvision.transforms.ToTensor()
            ]),
            torchvision.transforms.ToTensor()
        )
        return p

    def get_model_params(self):
        rblur_p = GaussianBlurLayer.ModelParams(GaussianBlurLayer, std=1.5)
        resnet_p = WideResnet.ModelParams(WideResnet, CommonModelParams(self.input_size, 10), num_classes=10, 
                                            normalization_layer_params=NormalizationLayer.get_params(), depth=self.depth, 
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rblur_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-5, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=352, pct_start=0.2, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128
        )

class Cifar10GaussianBlurCyclicLRAutoAugmentWideResNet2x22(Cifar10GaussianBlurCyclicLRAutoAugmentWideResNet1x22):
    widen_factor = 2

class Cifar10GaussianBlurCyclicLRAutoAugmentWideResNet4x22(Cifar10GaussianBlurCyclicLRAutoAugmentWideResNet1x22):
    widen_factor = 4

class Cifar10GaussianBlurCyclicLRAutoAugmentWideResNet8x22(Cifar10GaussianBlurCyclicLRAutoAugmentWideResNet1x22):
    widen_factor = 8

class Cifar10NoisyGaussianBlurCyclicLRAutoAugmentWideResNet4x22(Cifar10GaussianBlurCyclicLRAutoAugmentWideResNet4x22):
    noise_std = 0.0625
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = GaussianBlurLayer.ModelParams(GaussianBlurLayer, std=1.5)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = WideResnet.ModelParams(WideResnet, CommonModelParams(self.input_size, 10), num_classes=10, 
                                            normalization_layer_params=NormalizationLayer.get_params(), depth=self.depth, 
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p