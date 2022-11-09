import torchvision
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import NormalizationLayer
from adversarialML.biologically_inspired_models.src.models import (
    CommonModelParams, GeneralClassifier, SequentialLayers, XResNet34, XResNet18, XResNet50, WideResnet,
    ActivationLayer, BatchNorm2DLayer)
from adversarialML.biologically_inspired_models.src.retina_preproc import (
    RetinaBlurFilter, RetinaNonUniformPatchEmbedding, RetinaWarp, GaussianBlurLayer)
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
from adversarialML.biologically_inspired_models.src.imagenet_mlp_mixer_tasks_commons import get_basic_mlp_mixer_params
from adversarialML.biologically_inspired_models.src.vit_models import ViTClassifier
class Cifar10CyclicLRAutoAugmentWideResNet4x22(AbstractTask):
    input_size = [3, 32, 32]
    widen_factor = 4
    depth = 22
    def get_dataset_params(self):
        p = get_cifar10_params(num_train=50_000, num_test=10_000)
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
        resnet_p = WideResnet.ModelParams(WideResnet, CommonModelParams(self.input_size, 10), num_classes=10, 
                                            normalization_layer_params=NormalizationLayer.get_params(), depth=self.depth, 
                                            widen_factor=self.widen_factor)
        return resnet_p

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

class Ecoset10CyclicLRRandAugmentXResNet2x18(AbstractTask):
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
        return XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)

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

class Ecoset10RandAugmentMLPMixerS16(AbstractTask):
    input_width = 224
    def get_dataset_params(self) :
        p = get_ecoset10_params(train_transforms=[
                torchvision.transforms.Resize(self.input_width),
                torchvision.transforms.RandomCrop(self.input_width),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandAugment(magnitude=15)
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.input_width),
                torchvision.transforms.CenterCrop(self.input_width),
            ])
        return p
    
    def get_model_params(self):
        return get_basic_mlp_mixer_params([3,self.input_width,self.input_width], 10, 16, 512, 2048, 256, nn.GELU, 0., 8)
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            AdamOptimizerConfig(weight_decay=5e-5),
            OneCycleLRConfig(max_lr=0.001, epochs=nepochs, steps_per_epoch=375, pct_start=0.2, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128)

class Ecoset10RandAugmentViTB16(AbstractTask):
    input_width = 224
    def get_dataset_params(self) :
        p = get_ecoset10_params(train_transforms=[
                torchvision.transforms.Resize(self.input_width),
                torchvision.transforms.RandomCrop(self.input_width),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandAugment(magnitude=15)
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.input_width),
                torchvision.transforms.CenterCrop(self.input_width),
            ])
        return p
    
    def get_model_params(self):
        return ViTClassifier.ModelParams(ViTClassifier, num_labels=10)
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            AdamOptimizerConfig(weight_decay=5e-5),
            OneCycleLRConfig(max_lr=0.001, epochs=nepochs, steps_per_epoch=375, pct_start=0.2, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128)

class Ecoset10RandAugmentViTCustomSmall(Ecoset10RandAugmentViTB16):
    def get_model_params(self):
        return ViTClassifier.ModelParams(ViTClassifier, num_labels=10, hidden_size=512, num_hidden_layers=8, intermediate_size=2048, num_attention_heads=8)

class Ecoset100CyclicLRRandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
    def get_dataset_params(self) :
        p = get_ecoset100folder_params(num_train=500_000, train_transforms=[
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
        return XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 100), num_classes=100,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 40
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.2, epochs=nepochs, steps_per_epoch=1839, pct_start=0.2, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=256
        )

class EcosetCyclicLRRandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
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
        return XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 565), num_classes=565,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 25
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            # OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=1839, pct_start=0.05, anneal_strategy='linear'),
            # OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=5632, pct_start=0.1, anneal_strategy='linear'),
            OneCycleLRConfig(max_lr=0.2, epochs=nepochs, steps_per_epoch=5632, pct_start=0.2, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128
        )