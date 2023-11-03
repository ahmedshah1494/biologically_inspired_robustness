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

from task_utils import get_ecoset100folder_params, get_ecoset10_params, get_ecoset10folder_params

class Cifar10CyclicLRAutoAugmentWideResNet4x22(AbstractTask):
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
        return WideResnet.ModelParams(WideResnet, CommonModelParams([3, 32, 32], 10), num_classes=10, normalization_layer_params=NormalizationLayer.get_params(), depth=22, widen_factor=4)

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

class Cifar10RetinaWarpCyclicLRAutoAugmentWideResNet4x22(AbstractTask):
    input_size = [3, 32, 32]
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
        rblur_p = RetinaWarp.ModelParams(RetinaWarp, self.input_size, batch_size=32)
        resnet_p = WideResnet.ModelParams(WideResnet, CommonModelParams(self.input_size, 10), num_classes=10, normalization_layer_params=NormalizationLayer.get_params(), depth=22, widen_factor=4)
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
            OneCycleLRConfig(max_lr=1., epochs=nepochs, steps_per_epoch=352, pct_start=0.2),
            logdir=LOGDIR, batch_size=128
        )

class Cifar10RetinaBlurCyclicLRAutoAugmentWideResNet4x22(AbstractTask):
    input_size = [3, 32, 32]
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
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12)
        resnet_p = WideResnet.ModelParams(WideResnet, CommonModelParams(self.input_size, 10), num_classes=10, normalization_layer_params=NormalizationLayer.get_params(), depth=22, widen_factor=4)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rblur_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 30
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                ),
                AdversarialParams(testing_attack_params=get_apgd_inf_params([0, 0.008, 0.016, 0.024, 0.032], 50, 10))
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-5, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=1., epochs=nepochs, steps_per_epoch=352),
            logdir=LOGDIR, batch_size=128
        )

class Cifar10GaussianBlurCyclicLRAutoAugmentWideResNet4x22(AbstractTask):
    input_size = [3, 32, 32]
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
        resnet_p = WideResnet.ModelParams(WideResnet, CommonModelParams(self.input_size, 10), num_classes=10, normalization_layer_params=NormalizationLayer.get_params(), depth=22, widen_factor=4)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rblur_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 30
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                ),
                AdversarialParams(testing_attack_params=get_apgd_inf_params([0, 0.008, 0.016, 0.024, 0.032], 50, 10))
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-5, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=1., epochs=nepochs, steps_per_epoch=352),
            logdir=LOGDIR, batch_size=128
        )

class Imagenet100_64CyclicLRAutoAugmentXResNet18(AbstractTask):
    def get_dataset_params(self) :
        p = get_imagenet100_64_params(train_transforms=[
            torchvision.transforms.RandomCrop(64, 8),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.AutoAugment()
        ])
        return p
    
    def get_model_params(self):
        return XResNet18.ModelParams(XResNet18, CommonModelParams([3, 64, 64], 100), num_classes=100)

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 100
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            CyclicLRConfig(base_lr=1e-6, max_lr=0.2, step_size_up=1000*nepochs//8, step_size_down=1000*nepochs//8, mode='triangular2'),
            logdir=LOGDIR,
        )

class Imagenet100_64CyclicLR2AutoAugmentXResNet18(AbstractTask):
    def get_dataset_params(self) :
        p = get_imagenet100_64_params(train_transforms=[
            torchvision.transforms.RandomCrop(64, 8),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.AutoAugment()
        ])
        return p
    
    def get_model_params(self):
        return XResNet18.ModelParams(XResNet18, CommonModelParams([3, 64, 64], 100), num_classes=100)

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 100
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            CyclicLRConfig(base_lr=1e-5, max_lr=0.2, step_size_up=1000*5, step_size_down=1000*10, mode='exp_range', gamma=0.999947),
            logdir=LOGDIR, num_trainings=4
        )

class Imagenet100_64CyclicLRTr2AutoAugmentXResNet18(AbstractTask):
    def get_dataset_params(self) :
        p = get_imagenet100_64_params(train_transforms=[
            torchvision.transforms.RandomCrop(64, 8),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.AutoAugment()
        ])
        return p
    
    def get_model_params(self):
        return XResNet18.ModelParams(XResNet18, CommonModelParams([3, 64, 64], 100), num_classes=100)

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 200
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=30, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            CyclicLRConfig(base_lr=0.01, max_lr=0.1, step_size_up=1000*5, step_size_down=1000*10, mode='triangular2'),
            logdir=LOGDIR, num_trainings=4
        )

class Imagenet100_64RetinaBlurCyclicLR2AutoAugmentXResNet18(AbstractTask):
    input_size = [3, 64, 64]
    def get_dataset_params(self) :
        p = get_imagenet100_64_params(train_transforms=[
            torchvision.transforms.RandomCrop(64, 8),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.AutoAugment()
        ])
        return p
    
    def get_model_params(self):
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12)
        norm_p = NormalizationLayer.get_params()
        filter_p = SequentialLayers.ModelParams(SequentialLayers, [rblur_p, norm_p], 
                                            CommonModelParams(input_size=self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 100), num_classes=100, normalize_input=False)
        p = GeneralClassifier.ModelParams(GeneralClassifier, filter_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 100
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=30, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            CyclicLRConfig(base_lr=1e-5, max_lr=0.2, step_size_up=1000*5, step_size_down=1000*10, mode='exp_range', gamma=0.999947),
            logdir=LOGDIR, num_trainings=5
        )

class Imagenet100_64RetinaBlurCyclicLR200EpochsAutoAugmentXResNet18(AbstractTask):
    input_size = [3, 64, 64]
    def get_dataset_params(self) :
        p = get_imagenet100_64_params(train_transforms=[
            torchvision.transforms.RandomCrop(64, 8),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.AutoAugment()
        ])
        return p
    
    def get_model_params(self):
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12)
        norm_p = NormalizationLayer.get_params()
        filter_p = SequentialLayers.ModelParams(SequentialLayers, [rblur_p, norm_p], 
                                            CommonModelParams(input_size=self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 100), num_classes=100, normalize_input=False)
        p = GeneralClassifier.ModelParams(GeneralClassifier, filter_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 210
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=30, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            CyclicLRConfig(base_lr=1e-5, max_lr=0.2, step_size_up=1000*5, step_size_down=1000*10, mode='exp_range', gamma=0.999947),
            # CyclicLRConfig(base_lr=1e-4, max_lr=0.2, step_size_up=1000*5, step_size_down=1000*10, mode='exp_range', gamma=0.999985),
            logdir=LOGDIR, num_trainings=5
        )

class Imagenet100_64RetinaBlurStdx2CyclicLRAutoAugmentXResNet18(AbstractTask):
    input_size = [3, 64, 64]
    def get_dataset_params(self) :
        p = get_imagenet100_64_params(train_transforms=[
            torchvision.transforms.RandomCrop(64, 8),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.AutoAugment()
        ])
        return p
    
    def get_model_params(self):
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.24, 
                                                rod_std=0.18, max_rod_density=0.24)
        norm_p = NormalizationLayer.get_params()
        filter_p = SequentialLayers.ModelParams(SequentialLayers, [rblur_p, norm_p], 
                                            CommonModelParams(input_size=self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 100), num_classes=100, normalize_input=False)
        p = GeneralClassifier.ModelParams(GeneralClassifier, filter_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 210
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=30, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            CyclicLRConfig(base_lr=1e-5, max_lr=0.2, step_size_up=1000*5, step_size_down=1000*10, mode='exp_range', gamma=0.999947),
            # CyclicLRConfig(base_lr=1e-4, max_lr=0.2, step_size_up=1000*5, step_size_down=1000*10, mode='exp_range', gamma=0.999985),
            logdir=LOGDIR, num_trainings=5
        )

class Imagenet100_64RetinaNonUniformPatchEmbeddingCyclicLRAutoAugmentXResNet18(AbstractTask):
    input_size = [3, 64, 64]
    def get_dataset_params(self) :
        p = get_imagenet100_64_params(train_transforms=[
            torchvision.transforms.RandomCrop(64, 8),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.AutoAugment()
        ])
        return p
    
    def get_model_params(self):
        norm_p = NormalizationLayer.get_params()
        retina_p = RetinaNonUniformPatchEmbedding.ModelParams(RetinaNonUniformPatchEmbedding, self.input_size, hidden_size=32,
                                                                conv_stride=1, conv_padding='same', place_crop_features_in_grid=True,
                                                                normalization_layer_params=norm_p)
        bn_p = BatchNorm2DLayer.ModelParams(BatchNorm2DLayer, 32)
        relu_p = ActivationLayer.ModelParams(ActivationLayer, nn.ReLU)
        filter_p = SequentialLayers.ModelParams(SequentialLayers, [retina_p, bn_p, relu_p],
                                            CommonModelParams(input_size=self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([32]+self.input_size[1:], 100), num_classes=100, normalize_input=False)
        p = GeneralClassifier.ModelParams(GeneralClassifier, filter_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 210
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=30, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            CyclicLRConfig(base_lr=1e-5, max_lr=0.2, step_size_up=1000*5, step_size_down=1000*10, mode='exp_range', gamma=0.999947),
            logdir=LOGDIR, num_trainings=5
        )

class Imagenet100_81RetinaNonUniformPatchEmbeddingCyclicLRAutoAugmentXResNet18(AbstractTask):
    input_size = [3, 81, 81]
    retina_dim = 32
    def get_dataset_params(self) :
        p = get_imagenet100_81_params(train_transforms=[
            torchvision.transforms.RandomCrop(81, 8),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.AutoAugment()
        ])
        return p
    
    def get_model_params(self):
        norm_p = NormalizationLayer.get_params()
        retina_p = RetinaNonUniformPatchEmbedding.ModelParams(RetinaNonUniformPatchEmbedding, self.input_size, hidden_size=self.retina_dim,
                                                                conv_stride=1, conv_padding='same', place_crop_features_in_grid=True,
                                                                normalization_layer_params=norm_p, isobox_w=[3, 9, 27], rec_flds=[1, 3, 9, 27], 
                                                                visualize=False)
        bn_p = BatchNorm2DLayer.ModelParams(BatchNorm2DLayer, self.retina_dim)
        relu_p = ActivationLayer.ModelParams(ActivationLayer, nn.ReLU)
        filter_p = SequentialLayers.ModelParams(SequentialLayers, [retina_p, bn_p, relu_p],
                                            CommonModelParams(input_size=self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([self.retina_dim]+self.input_size[1:], 100), num_classes=100, normalize_input=False)
        p = GeneralClassifier.ModelParams(GeneralClassifier, filter_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 210
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=30, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            CyclicLRConfig(base_lr=1e-5, max_lr=0.2, step_size_up=1000*5, step_size_down=1000*10, mode='exp_range', gamma=0.999947),
            logdir=LOGDIR, num_trainings=5
        )

class Imagenet100_81RetinaNonUniformPatchEmbedding128DCyclicLRAutoAugmentXResNet18(AbstractTask):
    input_size = [3, 81, 81]
    retina_dim = 128
    def get_dataset_params(self) :
        p = get_imagenet100_81_params(train_transforms=[
            torchvision.transforms.RandomCrop(81, 8),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.AutoAugment()
        ])
        return p
    
    def get_model_params(self):
        norm_p = NormalizationLayer.get_params()
        retina_p = RetinaNonUniformPatchEmbedding.ModelParams(RetinaNonUniformPatchEmbedding, self.input_size, hidden_size=self.retina_dim,
                                                                conv_stride=1, conv_padding='same', place_crop_features_in_grid=True,
                                                                normalization_layer_params=norm_p, isobox_w=[3, 9, 27], rec_flds=[1, 3, 9, 27], 
                                                                visualize=False)
        bn_p = BatchNorm2DLayer.ModelParams(BatchNorm2DLayer, self.retina_dim)
        relu_p = ActivationLayer.ModelParams(ActivationLayer, nn.ReLU)
        filter_p = SequentialLayers.ModelParams(SequentialLayers, [retina_p, bn_p, relu_p],
                                            CommonModelParams(input_size=self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([self.retina_dim]+self.input_size[1:], 100), num_classes=100, normalize_input=False)
        p = GeneralClassifier.ModelParams(GeneralClassifier, filter_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 210
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=30, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            CyclicLRConfig(base_lr=1e-5, max_lr=0.2, step_size_up=1000*5, step_size_down=1000*10, mode='exp_range', gamma=0.999947),
            logdir=LOGDIR, num_trainings=5
        )

class Imagenet100_81RetinaNonUniformPatchEmbeddingCyclicLRAutoAugmentXResNet34(AbstractTask):
    input_size = [3, 81, 81]
    retina_dim = 32
    def get_dataset_params(self) :
        p = get_imagenet100_81_params(train_transforms=[
            torchvision.transforms.RandomCrop(81, 8),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.AutoAugment()
        ])
        return p
    
    def get_model_params(self):
        norm_p = NormalizationLayer.get_params()
        retina_p = RetinaNonUniformPatchEmbedding.ModelParams(RetinaNonUniformPatchEmbedding, self.input_size, hidden_size=self.retina_dim,
                                                                conv_stride=1, conv_padding='same', place_crop_features_in_grid=True,
                                                                normalization_layer_params=norm_p, isobox_w=[3, 9, 27], rec_flds=[1, 3, 9, 27], 
                                                                visualize=False)
        bn_p = BatchNorm2DLayer.ModelParams(BatchNorm2DLayer, self.retina_dim)
        relu_p = ActivationLayer.ModelParams(ActivationLayer, nn.ReLU)
        filter_p = SequentialLayers.ModelParams(SequentialLayers, [retina_p, bn_p, relu_p],
                                            CommonModelParams(input_size=self.input_size, activation=nn.Identity))
        resnet_p = XResNet34.ModelParams(XResNet34, CommonModelParams([self.retina_dim]+self.input_size[1:], 100), num_classes=100, normalize_input=False)
        p = GeneralClassifier.ModelParams(GeneralClassifier, filter_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 210
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=30, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            CyclicLRConfig(base_lr=1e-5, max_lr=0.2, step_size_up=1000*5, step_size_down=1000*10, mode='exp_range', gamma=0.999947),
            logdir=LOGDIR, num_trainings=5
        )

class Imagenet100_81CenteredRetinaNonUniformPatchEmbeddingCyclicLRAutoAugmentXResNet34(Imagenet100_81RetinaNonUniformPatchEmbeddingCyclicLRAutoAugmentXResNet34):
    def get_model_params(self):
        p = super().get_model_params()
        p.feature_model_params.layer_params[0].loc_mode='center'
        return p

class Imagenet100_81RetinaNonUniformPatchEmbedding128DCyclicLRAutoAugmentXResNet34(AbstractTask):
    input_size = [3, 81, 81]
    retina_dim = 128
    def get_dataset_params(self) :
        p = get_imagenet100_81_params(train_transforms=[
            torchvision.transforms.RandomCrop(81, 8),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.AutoAugment()
        ])
        return p
    
    def get_model_params(self):
        norm_p = NormalizationLayer.get_params()
        retina_p = RetinaNonUniformPatchEmbedding.ModelParams(RetinaNonUniformPatchEmbedding, self.input_size, hidden_size=self.retina_dim,
                                                                conv_stride=1, conv_padding='same', place_crop_features_in_grid=True,
                                                                normalization_layer_params=norm_p, isobox_w=[3, 9, 27], rec_flds=[1, 3, 9, 27], 
                                                                visualize=False)
        bn_p = BatchNorm2DLayer.ModelParams(BatchNorm2DLayer, self.retina_dim)
        relu_p = ActivationLayer.ModelParams(ActivationLayer, nn.ReLU)
        filter_p = SequentialLayers.ModelParams(SequentialLayers, [retina_p, bn_p, relu_p],
                                            CommonModelParams(input_size=self.input_size, activation=nn.Identity))
        resnet_p = XResNet34.ModelParams(XResNet34, CommonModelParams([self.retina_dim]+self.input_size[1:], 100), num_classes=100, normalize_input=False)
        p = GeneralClassifier.ModelParams(GeneralClassifier, filter_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 210
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=30, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            CyclicLRConfig(base_lr=1e-5, max_lr=0.2, step_size_up=1000*5, step_size_down=1000*10, mode='exp_range', gamma=0.999947),
            logdir=LOGDIR, num_trainings=5
        )

class Imagenet100_64RetinaBlurCyclicLR200EpochsAutoAugmentXResNet34(AbstractTask):
    input_size = [3, 64, 64]
    def get_dataset_params(self) :
        p = get_imagenet100_64_params(train_transforms=[
            torchvision.transforms.RandomCrop(64, 8),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.AutoAugment()
        ])
        return p
    
    def get_model_params(self):
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12)
        norm_p = NormalizationLayer.get_params()
        filter_p = SequentialLayers.ModelParams(SequentialLayers, [rblur_p, norm_p], 
                                            CommonModelParams(input_size=self.input_size, activation=nn.Identity))
        resnet_p = XResNet34.ModelParams(XResNet34, CommonModelParams(self.input_size, 100), num_classes=100, normalize_input=False)
        p = GeneralClassifier.ModelParams(GeneralClassifier, filter_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 210
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            # CyclicLRConfig(base_lr=1e-4, max_lr=0.2, step_size_up=1000*5, step_size_down=1000*10, mode='exp_range', gamma=0.999985),
            CyclicLRConfig(base_lr=1e-5, max_lr=0.2, step_size_up=1000*5, step_size_down=1000*10, mode='exp_range', gamma=0.999947),
            logdir=LOGDIR, num_trainings=5
        )

class ImagenetCyclicLRAutoAugmentXResNet18(AbstractTask):
    imgs_size = 112
    input_size = [3, imgs_size, imgs_size]
    def get_dataset_params(self) :
        p = get_imagenet_params(num_train=128, num_test=2, train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.RandomCrop(self.imgs_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.AutoAugment()
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        return p

    def get_model_params(self):
        return XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 1000), num_classes=1000, normalization_layer_params=NormalizationLayer.get_params())

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 3
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-5, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=9984, pct_start=0.2, anneal_strategy='linear', three_phase=True),
            logdir=LOGDIR, batch_size=64
        )

class EcosetCyclicLRAutoAugmentXResNet50(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    def get_dataset_params(self) :
        p = get_ecoset_params(num_train=176, num_test=8, train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.RandomCrop(self.imgs_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.AutoAugment()
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        return p

    def get_model_params(self):
        return XResNet50.ModelParams(XResNet50, CommonModelParams(self.input_size, 565), num_classes=565, normalization_layer_params=NormalizationLayer.get_params())

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 30
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-5, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=11264, pct_start=0.2, anneal_strategy='linear', three_phase=True),
            logdir=LOGDIR, batch_size=64
        )

class Ecoset10RetinaWarpCyclicLRRandAugmentXResNet18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
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
        rblur_p = RetinaWarp.ModelParams(RetinaWarp, self.input_size, batch_size=32)
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10, normalization_layer_params=NormalizationLayer.get_params())
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rblur_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 30
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-5, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=193, pct_start=0.2, anneal_strategy='linear', three_phase=True),
            logdir=LOGDIR, batch_size=256
        )

class Ecoset10CyclicLRAutoAugmentXResNet50(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    def get_dataset_params(self) :
        p = get_ecoset10_params(train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.RandomCrop(self.imgs_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.AutoAugment()
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        return p

    def get_model_params(self):
        return XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10, normalization_layer_params=NormalizationLayer.get_params())

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 30
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-5, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=193, pct_start=0.2, anneal_strategy='linear', three_phase=True),
            logdir=LOGDIR, batch_size=256
        )

class Ecoset10CyclicLRAutoAugmentXResNet18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    def get_dataset_params(self) :
        p = get_ecoset10_params(train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.RandomCrop(self.imgs_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.AutoAugment()
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        return p

    def get_model_params(self):
        return XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10, normalization_layer_params=NormalizationLayer.get_params())

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 30
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-5, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=193, pct_start=0.2, anneal_strategy='linear', three_phase=True),
            logdir=LOGDIR, batch_size=256
        )

class Ecoset10RetinaBlurCyclicLRAutoAugmentXResNet18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    def get_dataset_params(self) :
        p = get_ecoset10_params(train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.RandomCrop(self.imgs_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.AutoAugment()
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        return p

    def get_model_params(self):
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12)
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10, normalization_layer_params=NormalizationLayer.get_params())
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rblur_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 30
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-5, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=193, pct_start=0.2, anneal_strategy='linear', three_phase=True),
            logdir=LOGDIR, batch_size=256
        )

class Ecoset10RetinaBlurCyclicLRAutoAugmentXResNet50(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
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
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12)
        resnet_p = XResNet50.ModelParams(XResNet50, CommonModelParams(self.input_size, 10), num_classes=10, normalization_layer_params=NormalizationLayer.get_params())
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rblur_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 30
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-5, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=193, pct_start=0.2, anneal_strategy='linear', three_phase=True),
            logdir=LOGDIR, batch_size=256
        )

class Ecoset10GaussianBlurCyclicLRRandAugmentXResNet18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
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
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10, normalization_layer_params=NormalizationLayer.get_params())
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rblur_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 30
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-5, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=193, pct_start=0.2, anneal_strategy='linear', three_phase=True),
            logdir=LOGDIR, batch_size=256
        )

class Ecoset100CyclicLRAutoAugmentXResNet50(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    def get_dataset_params(self) :
        p = get_ecoset100folder_params(train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.RandomCrop(self.imgs_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.AutoAugment()
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        return p

    def get_model_params(self):
        return XResNet50.ModelParams(XResNet50, CommonModelParams(self.input_size, 100), num_classes=100, normalization_layer_params=NormalizationLayer.get_params())

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 30
        return BaseExperimentConfig(
            MixedPrecisionAdversarialTrainer.TrainerParams(MixedPrecisionAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-5, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=193, pct_start=0.2, anneal_strategy='linear', three_phase=True),
            logdir=LOGDIR, batch_size=256
        )

class ImagenetCyclicLRAutoAugmentXResNet50(AbstractTask):
    input_size = [3, 224, 224]
    def get_dataset_params(self) :
        p = get_imagenet_params(train_transforms=[
                torchvision.transforms.Resize(224),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.AutoAugment()
            ],
            test_transforms=[
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
            ])
        return p

    def get_model_params(self):
        return XResNet50.ModelParams(XResNet50, CommonModelParams(self.input_size, 1000), num_classes=1000, normalization_layer_params=NormalizationLayer.get_params())

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 30
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-5, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=9984, pct_start=0.2, anneal_strategy='linear', three_phase=True),
            logdir=LOGDIR, batch_size=32
        )

class ImagenetRetinaBlurCyclicLRAutoAugmentXResNet50(AbstractTask):
    input_size = [3, 224, 224]
    def get_dataset_params(self) :
        p = get_imagenet_params(train_transforms=[
                torchvision.transforms.Resize(224),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.AutoAugment()
            ],
            test_transforms=[
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
            ])
        return p

    def get_model_params(self):
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12)
        resnet_p = XResNet50.ModelParams(XResNet50, CommonModelParams(self.input_size, 1000), num_classes=1000, normalization_layer_params=NormalizationLayer.get_params())
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rblur_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 30
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-5, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=10000, pct_start=0.2, anneal_strategy='linear', three_phase=True),
            logdir=LOGDIR, batch_size=32
        )

class ImagenetFolderCyclicLRAutoAugmentXResNet50(AbstractTask):
    input_size = [3, 224, 224]
    def get_dataset_params(self) :
        p = get_imagenet_folder_params(train_transforms=[
                torchvision.transforms.Resize(224),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.AutoAugment()
            ],
            test_transforms=[
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
            ])
        return p

    def get_model_params(self):
        return XResNet50.ModelParams(XResNet50, CommonModelParams(self.input_size, 1000), num_classes=1000, normalization_layer_params=NormalizationLayer.get_params())

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 30
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-5, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=2451, pct_start=0.2, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128
        )

class ImagenetFolderRetinaBlurCyclicLRAutoAugmentXResNet50(AbstractTask):
    input_size = [3, 224, 224]
    def get_dataset_params(self) :
        p = get_imagenet_folder_params(train_transforms=[
                torchvision.transforms.Resize(224),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.AutoAugment()
            ],
            test_transforms=[
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
            ])
        return p

    def get_model_params(self):
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12)
        resnet_p = XResNet50.ModelParams(XResNet50, CommonModelParams(self.input_size, 1000), num_classes=1000, normalization_layer_params=NormalizationLayer.get_params())
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rblur_p, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 30
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-5, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=2451, pct_start=0.2, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128
        )