import torchvision
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import NormalizationLayer
from adversarialML.biologically_inspired_models.src.models import (
    CommonModelParams, GeneralClassifier, SequentialLayers, XResNet34, XResNet18, XResNet50, WideResnet, CORnetS,
    ActivationLayer, BatchNorm2DLayer, LogitAverageEnsembler, SupervisedContrastiveTrainingWrapper, IdentityLayer,
    XResNetClassifierWithReconstructionLoss, XResNetClassifierWithEnhancer, XResNetClassifierWithDeepResidualEnhancer)
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import LinearLayer
from adversarialML.biologically_inspired_models.src.retina_blur2 import RetinaBlurFilter as RBlur2
from adversarialML.biologically_inspired_models.src.retina_preproc import (
    RetinaBlurFilter, RetinaNonUniformPatchEmbedding, RetinaWarp, GaussianNoiseLayer)
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
from adversarialML.biologically_inspired_models.src.supconloss import \
    TwoCropTransform
from adversarialML.biologically_inspired_models.src.runners import TransferLearningExperimentConfig

class Ecoset10NoisyRetinaBlur2WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, 1600, 1600]
    widen_factor = 2
    widen_stem = False
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
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RBlur2.ModelParams(RBlur2, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='random_uniform',
                                                scale=12, min_res=33, max_res=400)
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True, widen_stem=self.widen_stem)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p, resnet_p], CommonModelParams(self.input_size, activation=nn.Identity))
        cp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.input_size, 10, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, cp)
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

class Ecoset10NoisyRetinaBlur2S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlur2WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    noise_std = 0.25

class Ecoset10NoisyRetinaBlurVF12002S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlur2S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    input_size = [3,1200,1200]

class Ecoset10NoisyRetinaBlurVF10002S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlur2S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    input_size = [3,1000,1000]

class Ecoset10NoisyRetinaBlurVF8002S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlur2S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    input_size = [3,800,800]

class Ecoset10NoisyRetinaBlur2WRandomScalesCyclicLR1e_1RandAugmentXResNet2x34(Ecoset10NoisyRetinaBlur2WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RBlur2.ModelParams(RBlur2, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='random_uniform',
                                                scale=12, min_res=33, max_res=400)
        resnet_p = XResNet34.ModelParams(XResNet34, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True, widen_stem=self.widen_stem)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p, resnet_p], CommonModelParams(self.input_size, activation=nn.Identity))
        cp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.input_size, 10, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, cp)
        return p

class Ecoset10NoisyRetinaBlur2S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18WideStem(Ecoset10NoisyRetinaBlur2S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    widen_stem = True

class Ecoset10NoisyRetinaBlur2S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet4x18WideStem(Ecoset10NoisyRetinaBlur2S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    widen_stem = True
    widen_factor = 4

    def get_experiment_params(self) -> BaseExperimentConfig:
        p = super().get_experiment_params()
        p.batch_size = 64
        return p

class Ecoset10NoisyRetinaBlur2WRandomScales400CyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlur2WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    imgs_size = 400
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
            logdir=LOGDIR, batch_size=64
        )

class Ecoset10NoisyRetinaBlur2S2500WRandomScalesWReconCyclicLR1e_1RandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, 1600, 1600]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.25
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
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RBlur2.ModelParams(RBlur2, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='random_uniform',
                                                scale=12, min_res=33, max_res=400)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNetClassifierWithReconstructionLoss.ModelParams(XResNetClassifierWithReconstructionLoss, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True, widen_stem=self.widen_stem,
                                            feature_layer_idx=2, preprocessing_params=rp, recon_wt=0.01, cls_wt=1.)
        # rp = XResNetReconstructionTrainingWrapper.ModelParams(XResNetReconstructionTrainingWrapper, CommonModelParams(self.input_size), rp)
        # cp = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
        #                                     widen_factor=self.widen_factor, setup_classification=True,
        #                                     setup_feature_extraction=False, widen_stem=self.widen_stem,
        #                                     feature_layer_idx=-5)
        # p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, cp)
        return resnet_p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=375, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128,
        )

class Ecoset10NoisyRetinaBlur2S2500WRandomScalesWRecon00CyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlur2S2500WRandomScalesWReconCyclicLR1e_1RandAugmentXResNet2x18):
    def get_model_params(self):
        p = super().get_model_params()
        p.recon_wt = 0.
        p.cls_wt = 1.
        return p

class Ecoset10NoisyRetinaBlur2S2500WRandomScalesWReconRandAugmentXResNetPretrain(Ecoset10NoisyRetinaBlur2S2500WRandomScalesWReconCyclicLR1e_1RandAugmentXResNet2x18):
    def get_model_params(self):
        p = super().get_model_params()
        p.recon_wt = 1.
        p.cls_wt = 0.
        return p

class Ecoset10NoisyRetinaBlur2S2500WRandomScalesWDeblurCyclicLR1e_1RandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, 1600, 1600]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.25
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
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RBlur2.ModelParams(RBlur2, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='random_uniform',
                                                scale=12, min_res=33, max_res=400)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNetClassifierWithEnhancer.ModelParams(XResNetClassifierWithEnhancer, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True, widen_stem=self.widen_stem,
                                            feature_layer_idx=2, preprocessing_params=rp, recon_wt=0.01, cls_wt=1.)
        return resnet_p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=375, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128,
        )

class Ecoset10NoisyRetinaBlur2S2500WRandomScalesWDeblurRandAugmentXResNetPretrain(Ecoset10NoisyRetinaBlur2S2500WRandomScalesWDeblurCyclicLR1e_1RandAugmentXResNet2x18):
    def get_model_params(self):
        p = super().get_model_params()
        p.recon_wt = 1.
        p.cls_wt = 0.
        return p

class Ecoset10NoisyRetinaBlur2S2500WRandomScalesWPretrainDeblurRandAugmentXResNet(Ecoset10NoisyRetinaBlur2S2500WRandomScalesWDeblurCyclicLR1e_1RandAugmentXResNet2x18):
    def get_model_params(self):
        p = super().get_model_params()
        p.recon_wt = 0.
        p.cls_wt = 1.
        return p
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=375, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128,
            seed_model_path=f'/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset10-0.0/Ecoset10NoisyRetinaBlur2S2500WRandomScalesWDeblurRandAugmentXResNetPretrain/1/checkpoints/model_checkpoint.pt',
            # # keys_to_skip_regex = r'(.*resnet.[6-7].*|classifier.classifier.*)',
            # # keys_to_freeze_regex = r'^(?!.*resnet.[6-7].*|classifier.classifier.*)'
            keys_to_freeze_regex = r'(.*multi_scale_conv.*|.*bnrelu.*|.*reconstructor.*)',
        )

class Ecoset10XResNetClassifierWithEnhancerRandAugmentXResNet2x18Pretrain(Ecoset10NoisyRetinaBlur2S2500WRandomScalesWPretrainDeblurRandAugmentXResNet):
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RBlur2.ModelParams(RBlur2, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='random_uniform',
                                                scale=12, min_res=33, max_res=400)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNetClassifierWithEnhancer.ModelParams(XResNetClassifierWithEnhancer, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True, widen_stem=self.widen_stem,
                                            feature_layer_idx=2, preprocessing_params=rp, recon_wt=0., cls_wt=1.,
                                            # no_reconstruction=True
                                            )
        return resnet_p

class Ecoset10NoisyRetinaBlur2S2500WRandomScalesDeepResidualDeblurCyclicLR1e_1RandAugmentXResNet2x18DeblurPretrain(AbstractTask):
    imgs_size = 224
    input_size = [3, 1600, 1600]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.25
    nclasses = 10
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
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RBlur2.ModelParams(RBlur2, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='random_uniform',
                                                scale=12, min_res=33, max_res=400)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNetClassifierWithDeepResidualEnhancer.ModelParams(XResNetClassifierWithDeepResidualEnhancer, CommonModelParams(self.input_size, self.nclasses), num_classes=self.nclasses,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True, widen_stem=self.widen_stem,
                                            feature_layer_idx=-1, preprocessing_params=rp, recon_wt=1., cls_wt=0.)
        return resnet_p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 20
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=375, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64,
            seed_model_path='/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset10-0.0/Ecoset10NoisyRetinaBlur2S2500WRandomScalesWDeblurRandAugmentXResNetPretrain/1/checkpoints/model_checkpoint.pt',
            keys_to_skip_regex = r'(.*convs.*|.*multi_scale_conv.*|.*bnrelu.*|.*upsample.*)',
            keys_to_freeze_regex = r'(.*resnet.*)',
        )
        # return BaseExperimentConfig(
        #     LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
        #         TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_loss',
        #             tracking_mode='min', scheduler_step_after_epoch=False
        #         )
        #     ),
        #     AdamOptimizerConfig(lr=0.2, weight_decay=5e-4),
        #     OneCycleLRConfig(max_lr=0.001, epochs=nepochs, steps_per_epoch=375*2, pct_start=0.1, anneal_strategy='linear'),
        #     logdir=LOGDIR, batch_size=64,
        # )
        
class Ecoset100NoisyRetinaBlur2S2500WRandomScalesDeepResidualDeblurCyclicLR1e_1RandAugmentXResNet2x18DeblurPretrain(Ecoset10NoisyRetinaBlur2S2500WRandomScalesDeepResidualDeblurCyclicLR1e_1RandAugmentXResNet2x18DeblurPretrain):
    nclasses = 100
    def get_dataset_params(self) :
        p = get_ecoset100folder_params(train_transforms=[
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
        # p.datafolder = f'{logdir_root}/ecoset/eval_dataset_dir'
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 15
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='train_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=1816, pct_start=0.1, anneal_strategy='linear'),
            # OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=1839, pct_start=0.05, anneal_strategy='cos'),
            logdir=LOGDIR, batch_size=64
        )

class Ecoset10NoisyRetinaBlur2S2500WRandomScalesDeepResidualDeblurCyclicLR1e_1RandAugmentXResNet2x18DeblurVGGFineTune(Ecoset10NoisyRetinaBlur2S2500WRandomScalesDeepResidualDeblurCyclicLR1e_1RandAugmentXResNet2x18DeblurPretrain):
    def get_model_params(self):
        p = super().get_model_params()
        p.perceptual_loss = True
        return p
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 10
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=375*2, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64,
            seed_model_path='/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset10-0.0/Ecoset10NoisyRetinaBlur2S2500WRandomScalesDeepResidualDeblurCyclicLR1e_1RandAugmentXResNet2x18DeblurPretrain/0/checkpoints/model_checkpoint.pt',
            keys_to_skip_regex=r'(.*resnet.*)',
            # keys_to_freeze_regex = r'(.*convs.*|.*multi_scale_conv.*|.*bnrelu.*|.*upsample.*)',
        )

class Ecoset10NoisyRetinaBlur2S2500WRandomScalesPretrainedDeepResidualDeblurCyclicLR1e_1RandAugmentXResNet2x18ClsTrain(Ecoset10NoisyRetinaBlur2S2500WRandomScalesDeepResidualDeblurCyclicLR1e_1RandAugmentXResNet2x18DeblurPretrain):
    def get_model_params(self):
        p = super().get_model_params()
        p.recon_wt = 0.
        p.cls_wt = 1.
        # p.no_reconstruction = True
        # p.preprocessing_params.no_blur = True
        # p.preprocessing_params.only_color = True
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 20
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=375, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128,
            seed_model_path='/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset10-0.0/Ecoset10NoisyRetinaBlur2S2500WRandomScalesDeepResidualDeblurCyclicLR1e_1RandAugmentXResNet2x18DeblurPretrain/0/checkpoints/model_checkpoint.pt',
            keys_to_skip_regex=r'(.*resnet.*)',
            keys_to_freeze_regex = r'(.*convs.*|.*multi_scale_conv.*|.*bnrelu.*|.*upsample.*)',
        )

class Ecoset10NoisyRetinaBlur2S2500WRandomScalesDeepResidualDeblurCyclicLR1e_1RandAugmentPretrainedXResNet2x18DeblurTrain(Ecoset10NoisyRetinaBlur2S2500WRandomScalesDeepResidualDeblurCyclicLR1e_1RandAugmentXResNet2x18DeblurPretrain):
    def get_model_params(self):
        p = super().get_model_params()
        p.recon_wt = 1.
        p.cls_wt = 1.
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=375, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128,
            seed_model_path='/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset10-0.0/Ecoset10NoisyRetinaBlur2S2500WRandomScalesWDeblurRandAugmentXResNetPretrain/1/checkpoints/model_checkpoint.pt',
            keys_to_skip_regex = r'(.*convs.*|.*multi_scale_conv.*|.*bnrelu.*|.*upsample.*)',
            keys_to_freeze_regex = r'(.*resnet.*)',
        )

class Ecoset10NoisyRetinaBlur2S2500WRandomScalesPretrainedDeepResidualDeblurCyclicLR1e_1RandAugmenPretrainedtXResNet2x18(Ecoset10NoisyRetinaBlur2S2500WRandomScalesDeepResidualDeblurCyclicLR1e_1RandAugmentXResNet2x18DeblurPretrain):
    def get_model_params(self):
        p = super().get_model_params()
        p.resnet_ckp_path = '/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset10-0.0/Ecoset10NoisyRetinaBlur2S2500WRandomScalesWDeblurRandAugmentXResNetPretrain/1/checkpoints/model_checkpoint.pt'
        p.reconstructor_ckp_path = '/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset10-0.0/Ecoset10NoisyRetinaBlur2S2500WRandomScalesDeepResidualDeblurCyclicLR1e_1RandAugmentXResNet2x18DeblurPretrain/0/checkpoints/model_checkpoint.pt'
        return p

class Ecoset10NoisyRetinaBlur2S2500WRandomScalesPretrainedDeepResidualDeblurCyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlur2S2500WRandomScalesDeepResidualDeblurCyclicLR1e_1RandAugmentXResNet2x18DeblurPretrain):
    def get_model_params(self):
        p = super().get_model_params()
        p.recon_wt = 0.
        p.cls_wt = 1.
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=375, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128,
            seed_model_path='/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset10-0.0/Ecoset10NoisyRetinaBlur2S2500WRandomScalesDeepResidualDeblurCyclicLR1e_1RandAugmentXResNet2x18DeblurPretrain/0/checkpoints/model_checkpoint.pt',
            # keys_to_skip_regex=r'(.*resnet.*)',
            keys_to_freeze_regex = r'(.*convs.*|.*multi_scale_conv.*|.*bnrelu.*|.*upsample.*)',
        )


class Ecoset10NoisyRetinaBlur2S2500WRandomScalesWPretainedReconCyclicLR1e_1RandAugmentXResNet2x18(Ecoset10NoisyRetinaBlur2S2500WRandomScalesWReconCyclicLR1e_1RandAugmentXResNet2x18):
    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=375, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128,
            seed_model_path=f'/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset10-0.0/Ecoset10NoisyRetinaBlur2S2500WRandomScalesWReconRandAugmentXResNetPretrain/0/checkpoints/model_checkpoint.pt',
            # # keys_to_skip_regex = r'(.*resnet.[6-7].*|classifier.classifier.*)',
            # # keys_to_freeze_regex = r'^(?!.*resnet.[6-7].*|classifier.classifier.*)'
            keys_to_freeze_regex = r'(.*resnet.[0-2].*)',
        )

class Ecoset10NoisyRetinaBlur2S2500WRandomScalesCyclicLR1e_1RandAugmentCORnetS(AbstractTask):
    imgs_size = 224
    input_size = [3, 1600, 1600]
    widen_factor = 2
    noise_std = 0.25
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
        # Pointing to a folder with only the test set, and some dummy train and val data. 
        # Use this on workhorse to avoid delay due to slow NFS.
        # p.datafolder = f'{logdir_root}/ecoset/eval_dataset_dir'
        return p

    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RBlur2.ModelParams(RBlur2, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                loc_mode='random_uniform_2', scale=11)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = CORnetS.ModelParams(CORnetS, CommonModelParams(self.input_size), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params())
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
            logdir=LOGDIR, batch_size=32
        )

class Ecoset100NoisyRetinaBlur2WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, 1600, 1600]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.125
    num_classes = 100
    def get_dataset_params(self) :
        p = get_ecoset100shards_params(num_test=10, train_transforms=[
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
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RBlur2.ModelParams(RBlur2, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='random_uniform',
                                                scale=12, min_res=33, max_res=400)
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, self.num_classes), num_classes=self.num_classes,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True, widen_stem=self.widen_stem)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p, resnet_p], CommonModelParams(self.input_size, activation=nn.Identity))
        cp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.input_size, self.num_classes, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, cp)
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
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=1830, pct_start=0.05, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64
        )

class Ecoset100NoisyRetinaBlur2S2500WRandomScalesCyclicLR1e_1RandAugmentCORnetS(AbstractTask):
    imgs_size = 224
    input_size = [3, 1600, 1600]
    widen_factor = 2
    noise_std = 0.25
    def get_dataset_params(self) :
        p = get_ecoset100folder_params(train_transforms=[
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
        # p.datafolder = f'{logdir_root}/ecoset/eval_dataset_dir'
        return p

    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RBlur2.ModelParams(RBlur2, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                loc_mode='random_uniform_2', scale=11)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = CORnetS.ModelParams(CORnetS, CommonModelParams(self.input_size), num_classes=100,
                                            normalization_layer_params=NormalizationLayer.get_params())
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
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=1816, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64 # 4 GPUs to get 256 bs
        )

class EcosetNoisyRetinaBlur2WRandomScalesCyclicLRRandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, 1600, 1600]
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
        # Pointing to a folder with only the test set, and some dummy train and val data. 
        # Use this on workhorse to avoid delay due to slow NFS.
        # p.datafolder = f'{logdir_root}/ecoset/eval_dataset_dir'
        return p

    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RBlur2.ModelParams(RBlur2, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                loc_mode='random_uniform_2', scale=11)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 565), num_classes=565,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, kernel_size=7)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 50
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            # OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=1839, pct_start=0.05, anneal_strategy='linear'),
            # OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=5632, pct_start=0.1, anneal_strategy='linear'),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=5632, pct_start=0.1, anneal_strategy='linear'),
            # OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=5632, pct_start=0.1, anneal_strategy='cos', div_factor=10, three_phase=True),
            logdir=LOGDIR, batch_size=64
        )

class EcosetNoisyRetinaBlur2S2500WRandomScalesCyclicRandAugmentViTB16(AbstractTask):
    imgs_size = 224
    input_size = [3, 1600, 1600]
    noise_std = 0.25
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
        rblur_p = RBlur2.ModelParams(RBlur2, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                loc_mode='random_uniform_2', scale=11)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        vit_p = ViTClassifier.ModelParams(ViTClassifier, num_labels=565)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, vit_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 100
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            AdamOptimizerConfig(weight_decay=5e-5),
            OneCycleLRConfig(max_lr=0.001, epochs=nepochs, steps_per_epoch=2816, pct_start=0.2, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128) # run with 4 GPUs to get batch size 512