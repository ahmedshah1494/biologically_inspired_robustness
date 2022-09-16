import torchvision
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import NormalizationLayer
from adversarialML.biologically_inspired_models.src.models import (
    CommonModelParams, GeneralClassifier, SequentialLayers, XResNet34, XResNet18,
    ActivationLayer, BatchNorm2DLayer)
from adversarialML.biologically_inspired_models.src.retina_preproc import (
    RetinaBlurFilter, RetinaNonUniformPatchEmbedding)
from adversarialML.biologically_inspired_models.src.supconloss import \
    TwoCropTransform
from adversarialML.biologically_inspired_models.src.trainers import MixedPrecisionAdversarialTrainer
from mllib.optimizers.configs import (AdamOptimizerConfig,
                                      CosineAnnealingWarmRestartsConfig,
                                      CyclicLRConfig, LinearLRConfig,
                                      ReduceLROnPlateauConfig,
                                      SequentialLRConfig, OneCycleLRConfig, SGDOptimizerConfig)
from mllib.runners.configs import BaseExperimentConfig, TrainingParams
from mllib.tasks.base_tasks import AbstractTask
from torch import nn
from task_utils import *

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
                                                rod_std=0.09, max_rod_density=0.12, kernel_size=18)
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
                                                rod_std=0.09, max_rod_density=0.12, kernel_size=18)
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
                                                rod_std=0.18, max_rod_density=0.24, kernel_size=18)
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
                                                rod_std=0.09, max_rod_density=0.12, kernel_size=18)
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