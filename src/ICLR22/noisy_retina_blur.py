import torchvision
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import NormalizationLayer
from adversarialML.biologically_inspired_models.src.models import (
    CommonModelParams, GeneralClassifier, SequentialLayers, XResNet34, XResNet18, XResNet50, WideResnet, CORnetS,
    ActivationLayer, BatchNorm2DLayer, LogitAverageEnsembler, SupervisedContrastiveTrainingWrapper, IdentityLayer,
    XResNetClassifierWithReconstructionLoss, XResNetClassifierWithEnhancer, XResNetClassifierWithDeepResidualEnhancer,
    MultiheadSelfAttentionEnsembler, LSTMEnsembler)
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import LinearLayer
from adversarialML.biologically_inspired_models.src.retina_blur2 import RetinaBlurFilter as RBlur2
from adversarialML.biologically_inspired_models.src.retina_preproc import (
    RetinaBlurFilter, RetinaNonUniformPatchEmbedding, RetinaWarp, GaussianNoiseLayer, RetinDoGBlurFilter)
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
# from adversarialML.biologically_inspired_models.src.imagenet_mlp_mixer_tasks_commons import get_basic_mlp_mixer_params
from adversarialML.biologically_inspired_models.src.vit_models import ViTClassifier
from adversarialML.biologically_inspired_models.src.supconloss import \
    TwoCropTransform
from adversarialML.biologically_inspired_models.src.runners import TransferLearningExperimentConfig

class Cifar10NoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22(AbstractTask):
    input_size = [3, 32, 32]
    widen_factor = 4
    depth = 22
    noise_std = 0.0625
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
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std, max_input_size=self.input_size)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform')
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = WideResnet.ModelParams(WideResnet, CommonModelParams(self.input_size, 10), num_classes=10, 
                                            normalization_layer_params=NormalizationLayer.get_params(), depth=self.depth, 
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
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

class Cifar10NoisyRetinaBlurCyclicLRAutoAugmentWideResNet4x22(AbstractTask):
    input_size = [3, 32, 32]
    widen_factor = 4
    depth = 22
    noise_std = 0.0625
    def get_dataset_params(self):
        p = get_cifar10_params(num_train=1_000)
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
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = WideResnet.ModelParams(WideResnet, CommonModelParams(self.input_size, 10), num_classes=10, 
                                            normalization_layer_params=NormalizationLayer.get_params(), depth=self.depth, 
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-5, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=352, pct_start=0.2, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128
        )

class Cifar10NoisyRetinaBlurOnlyColorWRandomScalesCyclicLRAutoAugmentWideResNet4x22(Cifar10NoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22):
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                only_color=True)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = WideResnet.ModelParams(WideResnet, CommonModelParams(self.input_size, 10), num_classes=10, 
                                            normalization_layer_params=NormalizationLayer.get_params(), depth=self.depth, 
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p

class Cifar10NoisyRetinaBlurNoBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22(Cifar10NoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22):
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                no_blur=True)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = WideResnet.ModelParams(WideResnet, CommonModelParams(self.input_size, 10), num_classes=10, 
                                            normalization_layer_params=NormalizationLayer.get_params(), depth=self.depth, 
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p

class Cifar10NoisyRetinaBlurS1250WRandomScalesCyclicLRAutoAugmentWideResNet4x22(AbstractTask):
    input_size = [3, 32, 32]
    widen_factor = 4
    depth = 22
    noise_std = 0.1250
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
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform')
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = WideResnet.ModelParams(WideResnet, CommonModelParams(self.input_size, 10), num_classes=10, 
                                            normalization_layer_params=NormalizationLayer.get_params(), depth=self.depth, 
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
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
class Cifar10NoisyRetinaBlurS2500WRandomScalesCyclicLRAutoAugmentWideResNet4x22(Cifar10NoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22):
    noise_std = 0.25

class Cifar10NoisyRetinaBlurS5000WRandomScalesCyclicLRAutoAugmentWideResNet4x22(Cifar10NoisyRetinaBlurWRandomScalesCyclicLRAutoAugmentWideResNet4x22):
    noise_std = 0.5

class Ecoset10NoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
    noise_std = 0.125
    def get_dataset_params(self) :
        p = get_ecoset10folder_params(train_transforms=[
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
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                use_1d_gkernels=True)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
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

class Ecoset10SupConNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    def get_dataset_params(self):
        p = super().get_dataset_params()
        trainT, testT = p.custom_transforms
        p.custom_transforms = (TwoCropTransform(trainT), testT)
        return p

    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform')
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p, resnet_p], CommonModelParams(self.input_size, activation=nn.Identity))
        cp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.input_size, 10, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, cp)
        p = SupervisedContrastiveTrainingWrapper.ModelParams(SupervisedContrastiveTrainingWrapper, p, IdentityLayer.get_params())
        return p
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=750, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64
        )

class Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    noise_std = 0.25

class Ecoset10NoisyRetinaBlurFovW8S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                use_1d_gkernels=True, min_bincount=self.imgs_size//50)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p
    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=375, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128
        )

class Ecoset10NoisyRetinaBlurFovW8S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18wMHSAFeatEnsembling(Ecoset10NoisyRetinaBlurFovW8S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
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
        mhsa_p = MultiheadSelfAttentionEnsembler.ModelParams(MultiheadSelfAttentionEnsembler, n=5, embed_dim=1024, num_heads=1, n_layers=1, n_repeats=1)
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                loc_mode='random_five_fixations', use_1d_gkernels=True, min_bincount=self.imgs_size//50)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, feature_ensembler_params=mhsa_p)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.2, epochs=nepochs, steps_per_epoch=375, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64
        )
    
class Ecoset10NoisyRetinaBlurFovW8S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18wMHSAFeatEnsemblingFineTuned(Ecoset10NoisyRetinaBlurFovW8S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18wMHSAFeatEnsembling):
    ckp_pth = f'/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset10-0.0/Ecoset10NoisyRetinaBlurFovW8S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18/0/checkpoints/epoch=59-step=22500.pt'
    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 10
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            # SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            AdamOptimizerConfig(weight_decay=5e-4),
            OneCycleLRConfig(max_lr=0.001, epochs=nepochs, steps_per_epoch=375, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64,
            seed_model_path=self.ckp_pth,
            keys_to_skip_regex = r'(classifier.classifier.*)',
            keys_to_freeze_regex = r'(.*resnet.[0-6].*)',
        )

class Ecoset10NeuronNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=1., neuronal_noise=True)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                use_1d_gkernels=True)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p
    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=375, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128
        )

class Ecoset10NoisyRetinaBlurS2500StdScale001WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    std_scale = .001
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                scale=self.std_scale)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p

class Ecoset10NoisyRetinaBlurS2500StdScale010WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlurS2500StdScale001WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    std_scale = .01
class Ecoset10NoisyRetinaBlurS2500StdScale025WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlurS2500StdScale001WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    std_scale = .025

class Ecoset10NoisyRetinaBlurS2500StdScale075WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlurS2500StdScale001WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    std_scale = .075

class Ecoset10NoisyRetinaBlurS5000WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    noise_std = 0.5

class Ecoset10NoisyRetinaBlurS2500CyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    noise_std = 0.25
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p

class Ecoset10NoisyRetinaBlurS2500OnlyColorWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    noise_std = 0.25
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                only_color=True)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p

class Ecoset10NoisyRetinaBlurS2500NoBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    noise_std = 0.25
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                no_blur=True)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p

class Ecoset10NoisyRetinaBlurWRandomScalesCyclicRandAugmentMLPMixerS16(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    noise_std = 0.125
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
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform')
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        mixer_p = get_basic_mlp_mixer_params(self.input_size, 10, 16, 512, 2048, 256, nn.GELU, 0., 8)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, mixer_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            AdamOptimizerConfig(weight_decay=5e-5),
            OneCycleLRConfig(max_lr=0.001, epochs=nepochs, steps_per_epoch=375, pct_start=0.2, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128)

class Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicRandAugmentMLPMixerS16(Ecoset10NoisyRetinaBlurWRandomScalesCyclicRandAugmentMLPMixerS16):
    noise_std = 0.25

class Ecoset10NoisyRetinaBlurWRandomScalesCyclicRandAugmentViTB16(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    noise_std = 0.125
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
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform')
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        vit_p = ViTClassifier.ModelParams(ViTClassifier, num_labels=10)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, vit_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            AdamOptimizerConfig(weight_decay=5e-5),
            OneCycleLRConfig(max_lr=0.001, epochs=nepochs, steps_per_epoch=375, pct_start=0.2, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128) 

class Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicRandAugmentViTB16(Ecoset10NoisyRetinaBlurWRandomScalesCyclicRandAugmentViTB16):
    noise_std = 0.25

class Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicRandAugmentViTCustomSmall(Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicRandAugmentViTB16):
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform')
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        vit_p = ViTClassifier.ModelParams(ViTClassifier, num_labels=10, hidden_size=512, num_hidden_layers=8, intermediate_size=2048, num_attention_heads=8)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, vit_p)
        return p

class Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
    noise_std = 0.125
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
        # Pointing to a folder with only the test set, and some dummy train and val data. 
        # Use this on workhorse to avoid delay due to slow NFS.
        p.datafolder = f'{logdir_root}/ecoset-100/eval_dataset_dir'
        return p

    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform')
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 100), num_classes=100,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
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
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=1839, pct_start=0.05, anneal_strategy='linear'),
            # OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=1839, pct_start=0.05, anneal_strategy='cos'),
            logdir=LOGDIR, batch_size=64
        )
    
class Ecoset100NoisyRetinaBlurMinBC4WRandomScalesCyclicLRRandAugmentXResNet2x18(Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18):
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                use_1d_gkernels=True, min_bincount=4, set_min_bin_to_1=True)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 100), num_classes=100,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p



class EcosetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
    noise_std = 0.125
    def get_dataset_params(self) :
        p = get_ecoset_folder_params(train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.RandomCrop(self.imgs_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandAugment(magnitude=15)
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        # Pointing to a folder with only the test set, and some dummy train and val data. 
        # Use this on workhorse to avoid delay due to slow NFS.
        p.datafolder = f'{logdir_root}/ecoset/eval_dataset_dir'
        return p

    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                use_1d_gkernels=True)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 565), num_classes=565,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p

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
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=5632, pct_start=0.1, anneal_strategy='cos', div_factor=10, three_phase=True),
            logdir=LOGDIR, batch_size=64
        )
    
class ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
    noise_std = 0.125
    def get_dataset_params(self) :
        p = get_imagenet_folder_params(num_train=1_275_000, train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.RandomCrop(self.imgs_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandAugment(magnitude=15)
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        # Pointing to a folder with only the test set, and some dummy train and val data. 
        # Use this on workhorse to avoid delay due to slow NFS.
        p.datafolder = f'{logdir_root}/imagenet/eval_dataset_dir'
        return p

    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                use_1d_gkernels=True)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 1000), num_classes=1000,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p

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
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=4916, pct_start=0.1, anneal_strategy='cos', div_factor=10, three_phase=True),
            logdir=LOGDIR, batch_size=64
        )

class ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18WLSTMFeatEnsemblingFT(ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18):
    ckp_pth = f'/ocean/projects/cis220031p/mshah1/biologically_inspired_models/iclr22_logs/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18/1/checkpoints/epoch=23-step=117984.pt'
    
    def get_dataset_params(self) :
        p = get_imagenet_folder_params(num_train=1_275_000, train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.RandomCrop(self.imgs_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandAugment(magnitude=15)
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        # Pointing to a folder with only the test set, and some dummy train and val data. 
        # Use this on workhorse to avoid delay due to slow NFS.
        # p.datafolder = f'{logdir_root}/imagenet/eval_dataset_dir'
        return p
    
    def get_model_params(self):
        lstmp = LSTMEnsembler.ModelParams(LSTMEnsembler, n=5, input_size=1024, hidden_size=1024)
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                loc_mode='random_five_fixations')
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 1000), num_classes=1000,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, feature_ensembler_params=lstmp)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 5
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            # SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            AdamOptimizerConfig(weight_decay=5e-4),
            OneCycleLRConfig(max_lr=0.001, epochs=nepochs, steps_per_epoch=4916//4, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64*4,
            seed_model_path=self.ckp_pth,
            keys_to_skip_regex = r'(classifier.classifier.*)',
            keys_to_freeze_regex = r'(.*resnet.*)',
        )

class ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18WMHSAFeatEnsemblingFT(ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18):
    ckp_pth = f'/ocean/projects/cis220031p/mshah1/biologically_inspired_models/iclr22_logs/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18/1/checkpoints/epoch=23-step=117984.pt'
    num_fixations = 5
    def get_dataset_params(self) :
        p = get_imagenet_folder_params(num_train=1_275_000, train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.RandomCrop(self.imgs_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandAugment(magnitude=15)
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        # Pointing to a folder with only the test set, and some dummy train and val data. 
        # Use this on workhorse to avoid delay due to slow NFS.
        # p.datafolder = f'{logdir_root}/imagenet/eval_dataset_dir'
        return p
    
    def get_model_params(self):
        mhsa_p = MultiheadSelfAttentionEnsembler.ModelParams(MultiheadSelfAttentionEnsembler, n=self.num_fixations, embed_dim=1024, num_heads=1, n_layers=1, n_repeats=1)
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                loc_mode='random_five_fixations', use_1d_gkernels=True)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 1000), num_classes=1000,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, feature_ensembler_params=mhsa_p)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 5
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            # SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            AdamOptimizerConfig(weight_decay=5e-4),
            OneCycleLRConfig(max_lr=0.001, epochs=nepochs, steps_per_epoch=4916//4, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64*4,
            seed_model_path=self.ckp_pth,
            keys_to_skip_regex = r'(classifier.classifier.*)',
            keys_to_freeze_regex = r'(.*resnet.*)',
        )
    
class ImagenetNoisyRetinaBlurBinCt4WRandomScalesCyclicLRRandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
    noise_std = 0.125
    def get_dataset_params(self) :
        p = get_imagenet_folder_params(num_train=1_275_000, train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.RandomCrop(self.imgs_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandAugment(magnitude=15)
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        # Pointing to a folder with only the test set, and some dummy train and val data. 
        # Use this on workhorse to avoid delay due to slow NFS.
        p.datafolder = f'{logdir_root}/imagenet/eval_dataset_dir'
        return p

    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                use_1d_gkernels=True, min_bincount=self.imgs_size//50)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 1000), num_classes=1000,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p

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
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=4916, pct_start=0.1, anneal_strategy='cos', div_factor=10, three_phase=True),
            logdir=LOGDIR, batch_size=64
        )