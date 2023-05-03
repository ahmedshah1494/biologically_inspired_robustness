import os

import torchvision
from adversarialML.biologically_inspired_models.src.fixation_prediction.models import (
    FixationHeatmapPredictionNetwork, FixationPredictionNetwork,
    RetinaFilterWithFixationPrediction,
    TiedBackboneRetinaFixationPreditionClassifier,
    DeepGazeII, DeepGazeIIE, DeepGazeIII)
from adversarialML.biologically_inspired_models.src.fixation_prediction.trainers import (
    ClickmeImportanceMapLightningAdversarialTrainer,
    FixationPointLightningAdversarialTrainer,
    RetinaFilterWithFixationPredictionLightningAdversarialTrainer)
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import (
    LinearLayer, NormalizationLayer)
from adversarialML.biologically_inspired_models.src.models import (
    CommonModelParams, ConvEncoder, ConvParams, GeneralClassifier,
    IdentityLayer, SequentialLayers, XResNet18, MultiheadSelfAttentionEnsembler)
from adversarialML.biologically_inspired_models.src.retina_blur2 import \
    RetinaBlurFilter as RBlur2
from adversarialML.biologically_inspired_models.src.retina_preproc import (
    GaussianNoiseLayer, RetinaBlurFilter, RetinaNonUniformPatchEmbedding,
    RetinaWarp)
from adversarialML.biologically_inspired_models.src.runners import \
    TransferLearningExperimentConfig
from adversarialML.biologically_inspired_models.src.supconloss import \
    TwoCropTransform
from adversarialML.biologically_inspired_models.src.task_utils import *
from adversarialML.biologically_inspired_models.src.trainers import \
    LightningAdversarialTrainer
from mllib.datasets.dataset_factory import SupportedDatasets
from mllib.optimizers.configs import (AdamOptimizerConfig,
                                      CosineAnnealingWarmRestartsConfig,
                                      CyclicLRConfig, LinearLRConfig,
                                      OneCycleLRConfig,
                                      ReduceLROnPlateauConfig,
                                      SequentialLRConfig, SGDOptimizerConfig)
from mllib.runners.configs import BaseExperimentConfig, TrainingParams
from mllib.tasks.base_tasks import AbstractTask
from torch import nn


class Ecoset10Train30KNoisyRetinaBlur2S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, 1600, 1600]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.25
    def get_dataset_params(self) :
        p = get_ecoset10folder_params(num_train=30_080, train_transforms=[
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
            OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=242, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128
        )

class Ecoset100Train400KNoisyRetinaBlur2WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, 1600, 1600]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.125
    num_classes = 100

    def get_dataset_params(self) :
        p = get_dataset_params(f'{logdir_root}/ecoset-100/400K-70K/400K/shards', SupportedDatasets.ECOSET100, num_train=32, num_test=10, train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.RandomCrop(self.imgs_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandAugment(magnitude=15)
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        # p = get_ecoset100folder_params(num_train=500000, train_transforms=[
        #         torchvision.transforms.Resize(self.imgs_size),
        #         torchvision.transforms.RandomCrop(self.imgs_size),
        #         torchvision.transforms.RandomHorizontalFlip(),
        #         torchvision.transforms.RandAugment(magnitude=15)
        #     ],
        #     test_transforms=[
        #         torchvision.transforms.Resize(self.imgs_size),
        #         torchvision.transforms.CenterCrop(self.imgs_size),
        #     ])
        # # Pointing to a folder with only the test set, and some dummy train and val data. 
        # # Use this on workhorse to avoid delay due to slow NFS.
        # p.datafolder = f'{logdir_root}/ecoset-100/eval_dataset_dir'
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
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=1563, pct_start=0.05, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64
        )

class Ecoset100Train400KNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18(Ecoset100Train400KNoisyRetinaBlur2WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18):
    imgs_size = 224
    input_size = [3, 224, 224]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.125
    num_classes = 100

    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='random_uniform')
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, self.num_classes), num_classes=self.num_classes,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True, widen_stem=self.widen_stem)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p, resnet_p], CommonModelParams(self.input_size, activation=nn.Identity))
        cp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.input_size, self.num_classes, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, cp)
        return p

class Ecoset10Train18KNoisyRetinaBlur2S2500WRandomScalesFixationPointPredictor(AbstractTask):
    imgs_size = 224
    input_size = [3, 1600, 1600]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.25
    def get_dataset_params(self) :
        p = get_dataset_params(f'{logdir_root}/ecoset-10/30K-18K/18K', SupportedDatasets.ECOSET10wFIXATIONMAPS_FOLDER, num_train=18000, num_test=1000,
        train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
                torchvision.transforms.RandomOrder([
                    torchvision.transforms.RandomEqualize(0.5),
                    torchvision.transforms.RandomAutocontrast(0.5),
                    torchvision.transforms.RandomGrayscale(0.5),
                    torchvision.transforms.RandomChoice(
                        [torchvision.transforms.RandomSolarize(0.5, 0.5),
                        torchvision.transforms.RandomInvert(0.5)], [0.5, 0.5]
                    ),
                    torchvision.transforms.RandomPosterize(4, 0.5),
                ])
            ],
        test_transforms=[
            torchvision.transforms.Resize(self.imgs_size),
            torchvision.transforms.CenterCrop(self.imgs_size),
        ])
        return p
    
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RBlur2.ModelParams(RBlur2, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale=None, loc_mode='random_in_image',
                                                scale=12, min_res=33, max_res=400)
        normp = NormalizationLayer.get_params()
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p, normp], CommonModelParams(self.input_size, activation=nn.Identity))
        convp = ConvEncoder.ModelParams(ConvEncoder,
            CommonModelParams(self.input_size, num_units=[64,64,64,64,1], dropout_p=0.),
            ConvParams([3,3,3,3,3], [2,2,2,2,2], [1,1,1,1,1])
        )
        p = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
                                                    CommonModelParams([3, self.imgs_size, self.imgs_size],),
                                                    convp, IdentityLayer.get_params(), 49, rp)
        return p
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 50
        return BaseExperimentConfig(
            FixationPointLightningAdversarialTrainer.TrainerParams(FixationPointLightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.05, epochs=nepochs, steps_per_epoch=141*4, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=32
        )

class Ecoset100Train70KNoisyRetinaBlur2WRandomScalesFixationPointPredictor(AbstractTask):
    imgs_size = 224
    input_size = [3, 1600, 1600]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.125
    num_classes = 100
    def get_dataset_params(self) :
        p = get_dataset_params(f'{logdir_root}/ecoset-100/400K-70K/70K', SupportedDatasets.ECOSET100wFIXATIONMAPS_FOLDER, num_train=70000, num_test=5000,
        train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
                torchvision.transforms.RandomOrder([
                    torchvision.transforms.RandomEqualize(0.5),
                    torchvision.transforms.RandomAutocontrast(0.5),
                    torchvision.transforms.RandomGrayscale(0.5),
                    torchvision.transforms.RandomChoice(
                        [torchvision.transforms.RandomSolarize(0.5, 0.5),
                        torchvision.transforms.RandomInvert(0.5)], [0.5, 0.5]
                    ),
                    torchvision.transforms.RandomPosterize(4, 0.5),
                ])
            ],
        test_transforms=[
            torchvision.transforms.Resize(self.imgs_size),
            torchvision.transforms.CenterCrop(self.imgs_size),
        ])
        return p
    
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RBlur2.ModelParams(RBlur2, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale=None, loc_mode='random_in_image',
                                                scale=12, min_res=33, max_res=400)
        normp = NormalizationLayer.get_params()
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p, normp], CommonModelParams(self.input_size, activation=nn.Identity))
        p = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
                                                    CommonModelParams([3, self.imgs_size, self.imgs_size],), 'unet', rp)
        return p
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 15
        return BaseExperimentConfig(
            FixationPointLightningAdversarialTrainer.TrainerParams(FixationPointLightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.05, epochs=nepochs, steps_per_epoch=1007, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64
        )

class Ecoset100Train70KNoisyRetinaBlurWRandomScalesFixationPointPredictor(Ecoset100Train70KNoisyRetinaBlur2WRandomScalesFixationPointPredictor):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale=None, loc_mode='random_uniform')
        normp = NormalizationLayer.get_params()
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, self.num_classes), num_classes=self.num_classes,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True, widen_stem=self.widen_stem)
        p = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
                                                    CommonModelParams([3, self.imgs_size, self.imgs_size],), 'deeplab3p', rp,
                                                    backbone_params=resnet_p, 
                                                    backbone_ckp_path='/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset100-0.0/Ecoset100Train400KNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18/0/checkpoints/model_checkpoint.pt',
                                                    freeze_backbone=True)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 15
        return BaseExperimentConfig(
            FixationPointLightningAdversarialTrainer.TrainerParams(FixationPointLightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.05, epochs=nepochs, steps_per_epoch=1007, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64
        )

class Ecoset100Train70KNoisyRetinaBlurWRandomScalesFixationPointPredictorMSELoss(Ecoset100Train70KNoisyRetinaBlur2WRandomScalesFixationPointPredictor):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale=None, loc_mode='random_uniform')
        normp = NormalizationLayer.get_params()
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, self.num_classes), num_classes=self.num_classes,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True, widen_stem=self.widen_stem)
        p = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
                                                    CommonModelParams([3, self.imgs_size, self.imgs_size],), 'deeplab3p', rp,
                                                    loss_fn='mse',
                                                    backbone_params=resnet_p, 
                                                    backbone_ckp_path='/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset100-0.0/Ecoset100Train400KNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18/0/checkpoints/model_checkpoint.pt',
                                                    freeze_backbone=True)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 15
        return BaseExperimentConfig(
            FixationPointLightningAdversarialTrainer.TrainerParams(FixationPointLightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=504, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128
        )

class Ecoset100Train70KNoisyRetinaBlurWRandomScalesFixationPointPredictorMSELossOS16(Ecoset100Train70KNoisyRetinaBlur2WRandomScalesFixationPointPredictor):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale=None, loc_mode='random_uniform')
        normp = NormalizationLayer.get_params()
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, self.num_classes), num_classes=self.num_classes,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True, widen_stem=self.widen_stem)
        p = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
                                                    CommonModelParams([3, self.imgs_size, self.imgs_size],), 'deeplab3p', rp,
                                                    loss_fn='mse',
                                                    backbone_params=resnet_p, 
                                                    backbone_ckp_path='/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset100-0.0/Ecoset100Train400KNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18/0/checkpoints/model_checkpoint.pt',
                                                    freeze_backbone=True,
                                                    llfeat_module_name='0',
                                                    llfeat_dim=32,
                                                    hlfeat_module_name='6',
                                                    hlfeat_dim=512)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 15
        return BaseExperimentConfig(
            FixationPointLightningAdversarialTrainer.TrainerParams(FixationPointLightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=1007, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64
        )

class Ecoset100Train70KNoisyRetinaBlurWRandomScalesFixationPointPredictorResNet18(Ecoset100Train70KNoisyRetinaBlur2WRandomScalesFixationPointPredictor):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]    
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='random_uniform')
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        p = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
                                                    CommonModelParams([3, self.imgs_size, self.imgs_size],), 'deeplab3p', rp,
                                                    backbone_params='resnet18', loss_fn='mse')
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 30
        return BaseExperimentConfig(
            FixationPointLightningAdversarialTrainer.TrainerParams(FixationPointLightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.05, epochs=nepochs, steps_per_epoch=504, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=128
        )
class ClickmeFixationHeatmapPredictor(AbstractTask):
    imgs_size = 224
    input_size = [3, 1600, 1600]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.25
    def get_dataset_params(self) :
        p = get_dataset_params(f'{logdir_root}/clickme/shards', SupportedDatasets.CLICKME, num_train=39, num_test=5,
        train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ],
        test_transforms=[
            torchvision.transforms.Resize(self.imgs_size),
            torchvision.transforms.CenterCrop(self.imgs_size),
        ])
        return p

    def get_model_params(self):
        normp = NormalizationLayer.get_params()
        # rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        p = FixationHeatmapPredictionNetwork.ModelParams(FixationHeatmapPredictionNetwork,
                                                    CommonModelParams([3, self.imgs_size, self.imgs_size],), normp)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 30
        return BaseExperimentConfig(
            ClickmeImportanceMapLightningAdversarialTrainer.TrainerParams(ClickmeImportanceMapLightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=9, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=True
                )
            ),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-4),
            # OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=4865, pct_start=0.1, anneal_strategy='linear'),
            ReduceLROnPlateauConfig(patience=3),
            logdir=LOGDIR, batch_size=64
        )

class ClickmeNoisyRetinaBlur2S2500WRandomScalesFixationHeatmapPredictor(ClickmeFixationHeatmapPredictor):
    imgs_size = 224
    input_size = [3, 1600, 1600]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.25
    def get_dataset_params(self) :
        p = get_dataset_params(f'{logdir_root}/clickme/shards', SupportedDatasets.CLICKME, num_train=39, num_test=5,
        train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ],
        test_transforms=[
            torchvision.transforms.Resize(self.imgs_size),
            torchvision.transforms.CenterCrop(self.imgs_size),
        ])
        return p

    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RBlur2.ModelParams(RBlur2, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='random_in_image',
                                                scale=12, min_res=33, max_res=400)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        p = FixationHeatmapPredictionNetwork.ModelParams(FixationHeatmapPredictionNetwork,
                                                    CommonModelParams([3, self.imgs_size, self.imgs_size],), rp)
        return p

    # def get_experiment_params(self) -> BaseExperimentConfig:
    #     nepochs = 30
    #     return BaseExperimentConfig(
    #         ClickmeImportanceMapLightningAdversarialTrainer.TrainerParams(ClickmeImportanceMapLightningAdversarialTrainer,
    #             TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=9, tracked_metric='val_loss',
    #                 tracking_mode='min', scheduler_step_after_epoch=True
    #             )
    #         ),
    #         AdamOptimizerConfig(lr=0.001, weight_decay=5e-4),
    #         # OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=4865, pct_start=0.1, anneal_strategy='linear'),
    #         ReduceLROnPlateauConfig(patience=3),
    #         logdir=LOGDIR, batch_size=64
    #     )

class ClickmeNoisyRetinaBlurS2500WRandomScalesFixationHeatmapPredictor(ClickmeFixationHeatmapPredictor):
    imgs_size = 224
    input_size = [3, 224, 224]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.25
    def get_dataset_params(self) :
        p = get_dataset_params(f'{logdir_root}/clickme/shards', SupportedDatasets.CLICKME, num_train=39, num_test=5,
        train_transforms=[
               torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
                torchvision.transforms.RandomOrder([
                    torchvision.transforms.RandomEqualize(0.5),
                    torchvision.transforms.RandomAutocontrast(0.5),
                    torchvision.transforms.RandomGrayscale(0.5),
                    torchvision.transforms.RandomChoice(
                        [torchvision.transforms.RandomSolarize(0.5, 0.5),
                        torchvision.transforms.RandomInvert(0.5)], [0.5, 0.5]
                    ),
                    torchvision.transforms.RandomPosterize(4, 0.5),
                ])
            ],
        test_transforms=[
            torchvision.transforms.Resize(self.imgs_size),
            torchvision.transforms.CenterCrop(self.imgs_size),
        ])
        return p

    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='random_uniform')
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        p = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
                                                    CommonModelParams([3, self.imgs_size, self.imgs_size],), 'deeplab3p', rp,
                                                    backbone_params='resnet18', loss_fn='mse')
        # p = FixationHeatmapPredictionNetwork.ModelParams(FixationHeatmapPredictionNetwork,
        #                                             CommonModelParams([3, self.imgs_size, self.imgs_size],), rp)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 30
        return BaseExperimentConfig(
            ClickmeImportanceMapLightningAdversarialTrainer.TrainerParams(ClickmeImportanceMapLightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=9, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False
                )
            ),
            # AdamOptimizerConfig(lr=0.001, weight_decay=5e-4),
            SGDOptimizerConfig(momentum=.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=4865//2, pct_start=0.1, anneal_strategy='linear'),
            # ReduceLROnPlateauConfig(patience=3),
            logdir=LOGDIR, batch_size=128
        )

class ClickmeNoisyRetinaBlurS1250WRandomScalesGALAFixationHeatmapPredictorXResNet(ClickmeNoisyRetinaBlurS2500WRandomScalesFixationHeatmapPredictor):
    noise_std = 0.125
    def get_dataset_params(self) :
        p = get_dataset_params(f'{logdir_root}/clickme/shards', SupportedDatasets.CLICKME, num_train=38, num_test=5,
        train_transforms=[
               torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
                torchvision.transforms.RandomOrder([
                    torchvision.transforms.RandomEqualize(0.5),
                    torchvision.transforms.RandomAutocontrast(0.5),
                    torchvision.transforms.RandomGrayscale(0.5),
                    torchvision.transforms.RandomChoice(
                        [torchvision.transforms.RandomSolarize(0.5, 0.5),
                        torchvision.transforms.RandomInvert(0.5)], [0.5, 0.5]
                    ),
                    torchvision.transforms.RandomPosterize(4, 0.5),
                ])
            ],
        test_transforms=[
            torchvision.transforms.Resize(self.imgs_size),
            torchvision.transforms.CenterCrop(self.imgs_size),
        ])
        return p

    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='center')
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True, widen_stem=self.widen_stem)
        rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p], CommonModelParams(self.input_size, activation=nn.Identity))
        p = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
                                                    CommonModelParams([3, self.imgs_size, self.imgs_size],), 'gala', rp,
                                                    backbone_params=resnet_p, loss_fn='mse')
        return p
    
    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 30
        return BaseExperimentConfig(
            ClickmeImportanceMapLightningAdversarialTrainer.TrainerParams(ClickmeImportanceMapLightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=9, tracked_metric='val_loss',
                    tracking_mode='min', scheduler_step_after_epoch=False
                )
            ),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-4),
            # SGDOptimizerConfig(momentum=.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.001, epochs=nepochs, steps_per_epoch=2432, pct_start=0.05, anneal_strategy='linear'),
            # ReduceLROnPlateauConfig(patience=3),
            logdir=LOGDIR, batch_size=128//N_GPUS
        )

class Ecoset10NoisyRetinaBlur2S2500WRandomScalesWClickmeFixationPredictionXResNet2x18(AbstractTask):
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
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='const',
                                                scale=12, min_res=33, max_res=400)
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True, widen_stem=self.widen_stem)
        fixp = FixationHeatmapPredictionNetwork.ModelParams(FixationHeatmapPredictionNetwork,
                                                    CommonModelParams([3, self.imgs_size, self.imgs_size],), IdentityLayer.get_params())
        # fixp = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
        #                                             CommonModelParams([3, self.imgs_size, self.imgs_size],),
        #                                             None, normp, 49, rnoise_p)
        retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,CommonModelParams(self.input_size), rnoise_p, rblur_p, fixp,
                                                        fixation_model_ckp='/share/workhorse3/mshah1/biologically_inspired_models/logs/clickme-0.0/ClickmeNoisyRetinaBlur2S2500WRandomScalesFixationHeatmapPredictor/3/checkpoints/model_checkpoint.pt'
                                                        # fixation_model_ckp='/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset10wfixationmaps_folder-0.0/Ecoset10Train18KNoisyRetinaBlur2S2500WRandomScalesFixationPointPredictor/1/checkpoints/model_checkpoint.pt'
        )
        rp = SequentialLayers.ModelParams(SequentialLayers, [IdentityLayer.get_params(), retinafixp, resnet_p], CommonModelParams(self.input_size, activation=nn.Identity))
        cp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.input_size, 10, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, cp)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=True
                )
            ),
            # SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            # OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=764, pct_start=0.1, anneal_strategy='linear'),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-4),
            ReduceLROnPlateauConfig(patience=3),
            logdir=LOGDIR, batch_size=64,
            seed_model_path=f'/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset10-0.0/Ecoset10NoisyRetinaBlur2S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18/0/checkpoints/model_checkpoint.pt',
            # # keys_to_skip_regex = r'(.*resnet.[6-7].*|classifier.classifier.*)',
            keys_to_freeze_regex = r'(.*resnet.*)',
        )

class Ecoset10Train30KNoisyRetinaBlur2S2500WRandomScalesWFixationPredictionXResNet2x18(AbstractTask):
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
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='const',
                                                scale=12, min_res=33, max_res=400)
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True, widen_stem=self.widen_stem)
        # fixp = FixationHeatmapPredictionNetwork.ModelParams(FixationHeatmapPredictionNetwork,
        #                                             CommonModelParams([3, self.imgs_size, self.imgs_size],), IdentityLayer.get_params())
        fixp = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
                                                    CommonModelParams([3, self.imgs_size, self.imgs_size],), 'unet', NormalizationLayer.get_params())
        retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,CommonModelParams(self.input_size), rnoise_p, rblur_p, fixp,
                                                        # fixation_model_ckp='/share/workhorse3/mshah1/biologically_inspired_models/logs/clickme-0.0/ClickmeNoisyRetinaBlur2S2500WRandomScalesFixationHeatmapPredictor/2/checkpoints/model_checkpoint.pt'
                                                        fixation_model_ckp='/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset10wfixationmaps_folder-0.0/Ecoset10Train18KNoisyRetinaBlur2S2500WRandomScalesFixationPointPredictor/4/checkpoints/model_checkpoint.pt'
        )
        rp = SequentialLayers.ModelParams(SequentialLayers, [IdentityLayer.get_params(), retinafixp, resnet_p], CommonModelParams(self.input_size, activation=nn.Identity))
        cp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.input_size, 10, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, cp)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=True
                )
            ),
            # SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            # OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=764, pct_start=0.1, anneal_strategy='linear'),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-4),
            ReduceLROnPlateauConfig(patience=3),
            logdir=LOGDIR, batch_size=64,
            seed_model_path=f'/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset10_folder-0.0/Ecoset10Train30KNoisyRetinaBlur2S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18/1/checkpoints/model_checkpoint.pt',
            # # keys_to_skip_regex = r'(.*resnet.[6-7].*|classifier.classifier.*)',
            keys_to_freeze_regex = r'(.*resnet.*)',
        )

class Ecoset100Train400KNoisyRetinaBlurWRandomScalesWFixationPredictionXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, 224, 224]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.125
    num_classes = 100
    ckp = '/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset100-0.0/Ecoset100Train400KNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18/0/checkpoints/model_checkpoint.pt'
    fpn_ckp = '/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset100wfixationmaps_folder-0.0/Ecoset100Train70KNoisyRetinaBlurWRandomScalesFixationPointPredictor/1/checkpoints/model_checkpoint.pt'
    def get_dataset_params(self) :
        p = get_ecoset100shards_params(train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.RandomCrop(self.imgs_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandAugment(magnitude=15)
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        # p = get_ecoset100folder_params(num_train=500000, train_transforms=[
        #         torchvision.transforms.Resize(self.imgs_size),
        #         torchvision.transforms.RandomCrop(self.imgs_size),
        #         torchvision.transforms.RandomHorizontalFlip(),
        #         torchvision.transforms.RandAugment(magnitude=15)
        #     ],
        #     test_transforms=[
        #         torchvision.transforms.Resize(self.imgs_size),
        #         torchvision.transforms.CenterCrop(self.imgs_size),
        #     ])
        # # Pointing to a folder with only the test set, and some dummy train and val data. 
        # # Use this on workhorse to avoid delay due to slow NFS.
        # p.datafolder = f'{logdir_root}/ecoset-100/eval_dataset_dir'
        return p

    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='random_uniform')
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, self.num_classes), num_classes=self.num_classes,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True, widen_stem=self.widen_stem)
        fixp = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
                                                    CommonModelParams([3, self.imgs_size, self.imgs_size],), 'deeplab3p', IdentityLayer.get_params(),
                                                    backbone_params=resnet_p, 
                                                    backbone_ckp_path=self.ckp,
                                                    freeze_backbone=True)
        retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,CommonModelParams(self.input_size), rnoise_p, rblur_p, fixp,
                                                        # fixation_model_ckp='/share/workhorse3/mshah1/biologically_inspired_models/logs/clickme-0.0/ClickmeNoisyRetinaBlur2S2500WRandomScalesFixationHeatmapPredictor/2/checkpoints/model_checkpoint.pt'
                                                        fixation_model_ckp=self.fpn_ckp
        )
        rp = SequentialLayers.ModelParams(SequentialLayers, [IdentityLayer.get_params(), retinafixp, resnet_p], CommonModelParams(self.input_size, activation=nn.Identity))
        # rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p, resnet_p], CommonModelParams(self.input_size, activation=nn.Identity))
        cp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.input_size, self.num_classes, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, cp)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=True
                )
            ),
            # SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            # OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=764, pct_start=0.1, anneal_strategy='linear'),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-4),
            ReduceLROnPlateauConfig(patience=3),
            logdir=LOGDIR, batch_size=64,
            seed_model_path=self.ckp,
            # # keys_to_skip_regex = r'(.*resnet.[6-7].*|classifier.classifier.*)',
            keys_to_freeze_regex = r'(.*resnet.*)',
        )

class Ecoset100PreTrain400KNoisyRetinaBlurWRandomScalesXResNet2x18WFixationPredictionMSELoss(Ecoset100Train400KNoisyRetinaBlurWRandomScalesWFixationPredictionXResNet2x18):
    fpn_ckp='/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset100wfixationmaps_folder-0.0/Ecoset100Train70KNoisyRetinaBlurWRandomScalesFixationPointPredictorMSELoss/0/checkpoints/model_checkpoint.pt'
    def get_dataset_params(self):
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

class Ecoset100PreTrain400KNoisyRetinaBlurWRandomScalesXResNet2x18WFixationPredictionMSELossOS16(Ecoset100Train400KNoisyRetinaBlurWRandomScalesWFixationPredictionXResNet2x18):
    fpn_ckp='/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset100wfixationmaps_folder-0.0/Ecoset100Train70KNoisyRetinaBlurWRandomScalesFixationPointPredictorMSELossOS16/0/checkpoints/model_checkpoint.pt'
    def get_dataset_params(self):
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
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='random_uniform')
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, self.num_classes), num_classes=self.num_classes,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True, widen_stem=self.widen_stem)
        fixp = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
                                                    CommonModelParams([3, self.imgs_size, self.imgs_size],), 'deeplab3p', IdentityLayer.get_params(),
                                                    backbone_params=resnet_p, 
                                                    backbone_ckp_path=self.ckp,
                                                    freeze_backbone=True,
                                                    llfeat_module_name='0',
                                                    llfeat_dim=32,
                                                    hlfeat_module_name='6',
                                                    hlfeat_dim=512)
        retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,CommonModelParams(self.input_size), rnoise_p, rblur_p, fixp,
                                                        # fixation_model_ckp='/share/workhorse3/mshah1/biologically_inspired_models/logs/clickme-0.0/ClickmeNoisyRetinaBlur2S2500WRandomScalesFixationHeatmapPredictor/2/checkpoints/model_checkpoint.pt'
                                                        fixation_model_ckp=self.fpn_ckp
        )
        rp = SequentialLayers.ModelParams(SequentialLayers, [IdentityLayer.get_params(), retinafixp, resnet_p], CommonModelParams(self.input_size, activation=nn.Identity))
        # rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p, resnet_p], CommonModelParams(self.input_size, activation=nn.Identity))
        cp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.input_size, self.num_classes, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, cp)
        return p

class Ecoset100PreTrain400KNoisyRetinaBlurWRandomScalesWClickmeFixationPredictionXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, 224, 224]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.125
    num_classes = 100
    ckp = '/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset100-0.0/Ecoset100Train400KNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18/0/checkpoints/model_checkpoint.pt'
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
        # p = get_ecoset100folder_params(num_train=500000, train_transforms=[
        #         torchvision.transforms.Resize(self.imgs_size),
        #         torchvision.transforms.RandomCrop(self.imgs_size),
        #         torchvision.transforms.RandomHorizontalFlip(),
        #         torchvision.transforms.RandAugment(magnitude=15)
        #     ],
        #     test_transforms=[
        #         torchvision.transforms.Resize(self.imgs_size),
        #         torchvision.transforms.CenterCrop(self.imgs_size),
        #     ])
        # # Pointing to a folder with only the test set, and some dummy train and val data. 
        # # Use this on workhorse to avoid delay due to slow NFS.
        # p.datafolder = f'{logdir_root}/ecoset-100/eval_dataset_dir'
        return p

    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='random_uniform')
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, self.num_classes), num_classes=self.num_classes,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True, widen_stem=self.widen_stem)
        fixp = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
                                                    CommonModelParams([3, self.imgs_size, self.imgs_size],), 'deeplab3p', IdentityLayer.get_params(),
                                                    backbone_params='resnet18', loss_fn='mse')
        retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,CommonModelParams(self.input_size), rnoise_p, rblur_p, fixp,
                                                        # fixation_model_ckp='/share/workhorse3/mshah1/biologically_inspired_models/logs/clickme-0.0/ClickmeNoisyRetinaBlur2S2500WRandomScalesFixationHeatmapPredictor/2/checkpoints/model_checkpoint.pt'
                                                        fixation_model_ckp='/share/workhorse3/mshah1/biologically_inspired_models/logs/clickme-0.0/ClickmeNoisyRetinaBlurS2500WRandomScalesFixationHeatmapPredictor/0/checkpoints/model_checkpoint.pt'
        )
        rp = SequentialLayers.ModelParams(SequentialLayers, [IdentityLayer.get_params(), retinafixp, resnet_p], CommonModelParams(self.input_size, activation=nn.Identity))
        # rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p, resnet_p], CommonModelParams(self.input_size, activation=nn.Identity))
        cp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.input_size, self.num_classes, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, cp)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 5
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.005, epochs=nepochs, steps_per_epoch=1849, pct_start=0.2, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64,
            seed_model_path=self.ckp,
            # # keys_to_skip_regex = r'(.*resnet.[6-7].*|classifier.classifier.*)',
            # keys_to_freeze_regex = r'(.*resnet.*)',
        )

class Ecoset100PreTrainedNoisyRetinaBlurWRandomScalesWClickmeFixationPredictionXResNet2x18FineTune(Ecoset100PreTrain400KNoisyRetinaBlurWRandomScalesWClickmeFixationPredictionXResNet2x18):
    ckp = '/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs/ecoset100_folder-0.0/ecoset100_folder-0.0/Ecoset100NoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18/0/checkpoints/model_checkpoint.pt'

    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='random_uniform')
        fixp = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
                                                    CommonModelParams([3, self.imgs_size, self.imgs_size],), 'deeplab3p', IdentityLayer.get_params(),
                                                    backbone_params='resnet18', loss_fn='mse')
        retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,CommonModelParams(self.input_size), rnoise_p, rblur_p, fixp,
                                                        fixation_model_ckp='/share/workhorse3/mshah1/biologically_inspired_models/logs/clickme-0.0/ClickmeNoisyRetinaBlurS2500WRandomScalesFixationHeatmapPredictor/0/checkpoints/model_checkpoint.pt'
        )
        rp = SequentialLayers.ModelParams(SequentialLayers, [IdentityLayer.get_params(), retinafixp], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, self.num_classes), num_classes=self.num_classes,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p

class Ecoset100NoisyRetinaBlurWRandomScalesWClickmeFixationPredictionXResNet2x18(Ecoset100PreTrainedNoisyRetinaBlurWRandomScalesWClickmeFixationPredictionXResNet2x18FineTune):
    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 40
        return BaseExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=1834, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64 # 4 GPUs to get 256 bs
        )

class Ecoset100PreTrainedNoisyRetinaBlurWRandomScalesWClickmeFixationPredictionXResNet2x18(Ecoset100PreTrainedNoisyRetinaBlurWRandomScalesWClickmeFixationPredictionXResNet2x18FineTune):
    pass

class Ecoset100PreTrain400KNoisyRetinaBlurWRandomScalesWFixationPredictionXResNet2x18FineTune(AbstractTask):
    imgs_size = 224
    input_size = [3, 224, 224]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.125
    num_classes = 100
    ckp = '/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset100-0.0/Ecoset100Train400KNoisyRetinaBlurWRandomScalesCyclicLR1e_1RandAugmentXResNet2x18/0/checkpoints/model_checkpoint.pt'
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
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='random_uniform')
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, self.num_classes), num_classes=self.num_classes,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True, widen_stem=self.widen_stem)
        fixp = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
                                                    CommonModelParams([3, self.imgs_size, self.imgs_size],), 'deeplab3p', IdentityLayer.get_params(),
                                                    backbone_params='resnet18', loss_fn='mse')
        retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,CommonModelParams(self.input_size), rnoise_p, rblur_p, fixp,
                                                        # fixation_model_ckp='/share/workhorse3/mshah1/biologically_inspired_models/logs/clickme-0.0/ClickmeNoisyRetinaBlur2S2500WRandomScalesFixationHeatmapPredictor/2/checkpoints/model_checkpoint.pt'
                                                        fixation_model_ckp='/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset100wfixationmaps_folder-0.0/Ecoset100Train70KNoisyRetinaBlurWRandomScalesFixationPointPredictor/0/checkpoints/model_checkpoint.pt'
        )
        rp = SequentialLayers.ModelParams(SequentialLayers, [IdentityLayer.get_params(), retinafixp, resnet_p], CommonModelParams(self.input_size, activation=nn.Identity))
        # rp = SequentialLayers.ModelParams(SequentialLayers, [rnoise_p, rblur_p, resnet_p], CommonModelParams(self.input_size, activation=nn.Identity))
        cp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.input_size, self.num_classes, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, cp)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 5
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.005, epochs=nepochs, steps_per_epoch=1834, pct_start=0.2, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64,
            seed_model_path=self.ckp,
            # # keys_to_skip_regex = r'(.*resnet.[6-7].*|classifier.classifier.*)',
            # keys_to_freeze_regex = r'(.*resnet.*)',
        )

class Ecoset100NoisyRetinaBlurWRandomScalesWTiedBackboneFixationPredictionXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, 224, 224]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.125
    num_classes = 100
    tau0 = 15
    tauN = 1
    def get_dataset_params(self) :
        # p = get_ecoset100shards_params(num_test=10, train_transforms=[
        #         torchvision.transforms.Resize(self.imgs_size),
        #         torchvision.transforms.RandomCrop(self.imgs_size),
        #         torchvision.transforms.RandomHorizontalFlip(),
        #         torchvision.transforms.RandAugment(magnitude=15)
        #     ],
        #     test_transforms=[
        #         torchvision.transforms.Resize(self.imgs_size),
        #         torchvision.transforms.CenterCrop(self.imgs_size),
        #     ])
        # p = get_ecoset100folder_params(num_train=500000, train_transforms=[
        #         torchvision.transforms.Resize(self.imgs_size),
        #         torchvision.transforms.RandomCrop(self.imgs_size),
        #         torchvision.transforms.RandomHorizontalFlip(),
        #         torchvision.transforms.RandAugment(magnitude=15)
        #     ],
        #     test_transforms=[
        #         torchvision.transforms.Resize(self.imgs_size),
        #         torchvision.transforms.CenterCrop(self.imgs_size),
        #     ])
        # # Pointing to a folder with only the test set, and some dummy train and val data. 
        # # Use this on workhorse to avoid delay due to slow NFS.
        # p.datafolder = f'{logdir_root}/ecoset-100/eval_dataset_dir'
        p = get_dataset_params(f'{logdir_root}/ecoset-100/400K-70K/70K', SupportedDatasets.ECOSET100wFIXATIONMAPS_FOLDER, num_train=70000, num_test=5000,
                train_transforms=[
                        torchvision.transforms.Resize(self.imgs_size),
                        torchvision.transforms.CenterCrop(self.imgs_size),
                    ],
                test_transforms=[
                    torchvision.transforms.Resize(self.imgs_size),
                    torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        return p

    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='random_uniform')
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, self.num_classes), num_classes=self.num_classes,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_classification=False,
                                            setup_feature_extraction=True, widen_stem=self.widen_stem)
        fixp = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
                                                    CommonModelParams([3, self.imgs_size, self.imgs_size],), 'deeplab3p', IdentityLayer.get_params(),
                                                    backbone_params=resnet_p, loss_fn='mse')
        retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,CommonModelParams(self.input_size), 
                                                                    rnoise_p, rblur_p, fixation_params=fixp, freeze_fixation_model=False, 
                                                                    target_downsample_factor=56, loc_sampling_temp=self.tau0)
        p = TiedBackboneRetinaFixationPreditionClassifier.ModelParams(TiedBackboneRetinaFixationPreditionClassifier, CommonModelParams(self.input_size, self.num_classes), 
                                                                        retinafixp, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 40
        steps_per_epoch = 1834
        gamma = (self.tauN/self.tau0)**(1/(nepochs*steps_per_epoch))
        return BaseExperimentConfig(
            RetinaFilterWithFixationPredictionLightningAdversarialTrainer.TrainerParams(RetinaFilterWithFixationPredictionLightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                ), loc_sampling_temp_decay_rate=gamma
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=steps_per_epoch, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64 # 4 GPUs to get 256 bs
        )

class Ecoset10NoisyRetinaBlurS2500WRandomScalesXResNet2x18WDeepGazeIIE(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.25
    ckp_pth = f'/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs/ecoset10-0.0/Ecoset10NoisyRetinaBlurS2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18/0/checkpoints/model_checkpoint.pt'
    def get_dataset_params(self) :
        p = get_ecoset10folder_params(train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.RandomCrop(self.imgs_size),
                torchvision.transforms.RandomHorizontalFlip()
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        return p

    def get_model_params(self):
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale=0, loc_mode='const',
                                                use_1d_gkernels=True, min_bincount=14, set_min_bin_to_1=False)
        fixp = DeepGazeIIE.ModelParams(DeepGazeIIE)
        retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,
                                                                        CommonModelParams(self.input_size), None, rblur_p, fixp,
                                                                        target_downsample_factor=16, apply_retina_before_fixation=False)
        rp = SequentialLayers.ModelParams(SequentialLayers, [IdentityLayer.get_params(), retinafixp], CommonModelParams(self.input_size, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=True
                )
            ),
            # SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            # OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=764, pct_start=0.1, anneal_strategy='linear'),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-4),
            ReduceLROnPlateauConfig(patience=3),
            logdir=LOGDIR, batch_size=64,
            seed_model_path=self.ckp_pth,
            # # keys_to_skip_regex = r'(.*resnet.[6-7].*|classifier.classifier.*)',
            # keys_to_freeze_regex = r'(.*resnet.*)',
        )

class Ecoset10NoisyRetinaBlurS2500WRandomScalesXResNet2x18FineTunedWDeepGazeIIE(Ecoset10NoisyRetinaBlurS2500WRandomScalesXResNet2x18WDeepGazeIIE):
    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 10
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            # SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            AdamOptimizerConfig(lr=1e-4, weight_decay=5e-4),
            OneCycleLRConfig(max_lr=1e-4, epochs=nepochs, steps_per_epoch=750, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=16,
            seed_model_path=self.ckp_pth,
            # # keys_to_skip_regex = r'(.*resnet.[6-7].*|classifier.classifier.*)',
            # keys_to_freeze_regex = r'(.*resnet.*)',
        )
    
class Ecoset10NoisyRetinaBlurFovW8S2500WRandomScalesXResNet2x18WDeepGazeIIE(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.25
    ckp_pth = f'/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset10-0.0/Ecoset10NoisyRetinaBlurFovW8S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18/0/checkpoints/epoch=59-step=22500.pt'
    def get_dataset_params(self) :
        p = get_ecoset10folder_params(train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.RandomCrop(self.imgs_size),
                torchvision.transforms.RandomHorizontalFlip()
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        return p

    def get_model_params(self):
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                use_1d_gkernels=True, min_bincount=self.imgs_size//50)
        fixp = DeepGazeIIE.ModelParams(DeepGazeIIE)
        retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,
                                                                        CommonModelParams(self.input_size), None, rblur_p, fixp,
                                                                        target_downsample_factor=16, apply_retina_before_fixation=True)
        rp = SequentialLayers.ModelParams(SequentialLayers, [IdentityLayer.get_params(), retinafixp], CommonModelParams(self.input_size, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=True
                )
            ),
            # SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            # OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=764, pct_start=0.1, anneal_strategy='linear'),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-4),
            ReduceLROnPlateauConfig(patience=3),
            logdir=LOGDIR, batch_size=64,
            seed_model_path=self.ckp_pth,
            # # keys_to_skip_regex = r'(.*resnet.[6-7].*|classifier.classifier.*)',
            # keys_to_freeze_regex = r'(.*resnet.*)',
        )
    
class Ecoset10NoisyRetinaBlurMinBC4S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18WDeepGazeIIEFixations(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
    noise_std = 0.25

    def get_dataset_params(self) :
        p = get_dataset_params(f'{logdir_root}/ecoset-10', SupportedDatasets.ECOSET10wFIXATIONMAPS_FOLDER, num_train=50000, num_test=1000,
        train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
                torchvision.transforms.RandomOrder([
                    torchvision.transforms.RandomEqualize(0.25),
                    torchvision.transforms.RandomAutocontrast(0.25),
                    torchvision.transforms.RandomGrayscale(0.25),
                    torchvision.transforms.RandomChoice(
                        [torchvision.transforms.RandomSolarize(0.5, 0.25),
                        torchvision.transforms.RandomInvert(0.25)], [0.5, 0.5]
                    ),
                    torchvision.transforms.RandomPosterize(4, 0.25),
                ])
            ],
        test_transforms=[
            torchvision.transforms.Resize(self.imgs_size),
            torchvision.transforms.CenterCrop(self.imgs_size),
        ],
        fixation_map_root=f'{logdir_root}/ecoset-10/fixation_maps/deepgaze2e/',
        fmap_transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.imgs_size),
            torchvision.transforms.CenterCrop(self.imgs_size)
        ])
        )
        return p
    
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                use_1d_gkernels=True, min_bincount=self.imgs_size//50)
        retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,
                                                                        CommonModelParams([4, self.imgs_size, self.imgs_size]), rnoise_p, rblur_p, IdentityLayer.get_params(),
                                                                        target_downsample_factor=16, apply_retina_before_fixation=False, salience_map_provided_as_input_channel=True,
                                                                        loc_sampling_temp=10.)
        rp = SequentialLayers.ModelParams(SequentialLayers, [IdentityLayer.get_params(), retinafixp], CommonModelParams([4, self.imgs_size, self.imgs_size], activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, [4, self.imgs_size, self.imgs_size], rp, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return BaseExperimentConfig(
            RetinaFilterWithFixationPredictionLightningAdversarialTrainer.TrainerParams(RetinaFilterWithFixationPredictionLightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            OneCycleLRConfig(max_lr=0.2, epochs=nepochs, steps_per_epoch=375, pct_start=0.1, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64
        )
    
class Ecoset10NoisyRetinaBlurMinBC4S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18WDeepGazeIIEFixations2(Ecoset10NoisyRetinaBlurMinBC4S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18WDeepGazeIIEFixations):
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, std=self.noise_std)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                use_1d_gkernels=True, min_bincount=self.imgs_size//50)
        retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,
                                                                        CommonModelParams([4, self.imgs_size, self.imgs_size]), rnoise_p, rblur_p, IdentityLayer.get_params(),
                                                                        target_downsample_factor=16, apply_retina_before_fixation=False, salience_map_provided_as_input_channel=True,
                                                                        random_fixation_prob=0.5)
        rp = SequentialLayers.ModelParams(SequentialLayers, [IdentityLayer.get_params(), retinafixp], CommonModelParams([4, self.imgs_size, self.imgs_size], activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = GeneralClassifier.ModelParams(GeneralClassifier, [4, self.imgs_size, self.imgs_size], rp, resnet_p)
        return p

class Ecoset10NoisyRetinaBlurFovW8S2500WRandomScalesXResNet2x18WDeepGazeII(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.25
    ckp_pth = f'/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset10-0.0/Ecoset10NoisyRetinaBlurFovW8S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18/0/checkpoints/epoch=59-step=22500.pt'
    def get_dataset_params(self) :
        p = get_ecoset10folder_params(train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.RandomCrop(self.imgs_size),
                torchvision.transforms.RandomHorizontalFlip()
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        return p

    def get_model_params(self):
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                use_1d_gkernels=True, min_bincount=self.imgs_size//50)
        # fixp = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
        #                                             CommonModelParams(self.input_size,), 'deepgaze2')
        fixp = DeepGazeII.ModelParams(DeepGazeII)
        retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,
                                                                        CommonModelParams(self.input_size), None, rblur_p, fixp,
                                                                        target_downsample_factor=16, apply_retina_before_fixation=True)
        rp = SequentialLayers.ModelParams(SequentialLayers, [IdentityLayer.get_params(), retinafixp], CommonModelParams(self.input_size, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=True
                )
            ),
            # SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            # OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=764, pct_start=0.1, anneal_strategy='linear'),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-4),
            ReduceLROnPlateauConfig(patience=3),
            logdir=LOGDIR, batch_size=64,
            seed_model_path=self.ckp_pth,
            # # keys_to_skip_regex = r'(.*resnet.[6-7].*|classifier.classifier.*)',
            # keys_to_freeze_regex = r'(.*resnet.*)',
        )

class Ecoset10NoisyRetinaBlurFovW8S2500WRandomScalesXResNet2x18WDeepGazeIIW5MHSAFeatEnsembling(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.25
    ckp_pth = f'/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset10-0.0/Ecoset10NoisyRetinaBlurFovW8S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18/0/checkpoints/epoch=59-step=22500.pt'
    def get_dataset_params(self) :
        p = get_ecoset10_params(train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.RandomCrop(self.imgs_size),
                torchvision.transforms.RandomHorizontalFlip()
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        return p

    def get_model_params(self):
        mhsa_p = MultiheadSelfAttentionEnsembler.ModelParams(MultiheadSelfAttentionEnsembler, n=5, embed_dim=1024, num_heads=8, n_layers=1)
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, feature_ensembler_params=mhsa_p)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                use_1d_gkernels=True, min_bincount=self.imgs_size//50)
        # fixp = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
        #                                             CommonModelParams(self.input_size,), 'deepgaze2')
        fixp = DeepGazeII.ModelParams(DeepGazeII)
        retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,
                                                                        CommonModelParams(self.input_size), None, rblur_p, fixp,
                                                                        target_downsample_factor=16, apply_retina_before_fixation=True,
                                                                        num_train_fixation_points=5, num_eval_fixation_points=5, 
                                                                        freeze_fixation_model=True, loc_sampling_temp=10., random_fixation_prob=0.25)
        rp = SequentialLayers.ModelParams(SequentialLayers, [IdentityLayer.get_params(), retinafixp], CommonModelParams(self.input_size, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, resnet_p)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 2
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=False
                )
            ),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-4),
            OneCycleLRConfig(max_lr=0.001, epochs=nepochs, steps_per_epoch=1502, pct_start=0.2, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=16,
            seed_model_path=self.ckp_pth,
            keys_to_skip_regex = r'(classifier.classifier.*)',
            keys_to_freeze_regex = r'(.*resnet.*)',
        )
    
class Ecoset10NoisyRetinaBlurFovW8S2500WRandomScalesXResNet2x18WDeepGazeIII(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.25
    ckp_pth = f'/share/workhorse3/mshah1/biologically_inspired_models/logs/ecoset10-0.0/Ecoset10NoisyRetinaBlurFovW8S2500WRandomScalesCyclicLR1e_1RandAugmentXResNet2x18/0/checkpoints/epoch=59-step=22500.pt'
    def get_dataset_params(self) :
        p = get_ecoset10folder_params(train_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.RandomCrop(self.imgs_size),
                torchvision.transforms.RandomHorizontalFlip()
            ],
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        return p

    def get_model_params(self):
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_feature_extraction=True,
                                            setup_classification=False)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                use_1d_gkernels=True, min_bincount=self.imgs_size//50)
        # fixp = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
        #                                             CommonModelParams(self.input_size,), 'deepgaze2')
        fixp = DeepGazeIII.ModelParams(DeepGazeIII)
        retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,
                                                                        CommonModelParams(self.input_size), None, rblur_p, fixp,
                                                                        target_downsample_factor=16, apply_retina_before_fixation=True)
        rp = SequentialLayers.ModelParams(SequentialLayers, [IdentityLayer.get_params(), retinafixp, resnet_p], CommonModelParams(self.input_size, activation=nn.Identity))
        cp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.input_size, 10, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, cp)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=True
                )
            ),
            # SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            # OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=764, pct_start=0.1, anneal_strategy='linear'),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-4),
            ReduceLROnPlateauConfig(patience=3),
            logdir=LOGDIR, batch_size=64,
            seed_model_path=self.ckp_pth,
            # # keys_to_skip_regex = r'(.*resnet.[6-7].*|classifier.classifier.*)',
            # keys_to_freeze_regex = r'(.*resnet.*)',
            prefix_map={'classifier.resnet':'feature_model.layers.2.resnet', 'classifier.classifier':'classifier.layer',}
        )
    
class ImagenetNoisyRetinaBlurWRandomScalesXResNet2x18WDeepGazeIII(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.125
    ckp_pth = f'/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18/1/checkpoints/model_checkpoint.pt'
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
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 1000), num_classes=1000,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_feature_extraction=True,
                                            setup_classification=False)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                use_1d_gkernels=True)
        # fixp = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
        #                                             CommonModelParams(self.input_size,), 'deepgaze2')
        fixp = DeepGazeIII.ModelParams(DeepGazeIII)
        retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,
                                                                        CommonModelParams(self.input_size), rnoise_p, rblur_p, fixp,
                                                                        target_downsample_factor=16, apply_retina_before_fixation=True)
        rp = SequentialLayers.ModelParams(SequentialLayers, [IdentityLayer.get_params(), retinafixp, resnet_p], CommonModelParams(self.input_size, activation=nn.Identity))
        cp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.input_size, 1000, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, cp)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=True
                )
            ),
            # SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            # OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=764, pct_start=0.1, anneal_strategy='linear'),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-4),
            ReduceLROnPlateauConfig(patience=3),
            logdir=LOGDIR, batch_size=64,
            seed_model_path=self.ckp_pth,
            # # keys_to_skip_regex = r'(.*resnet.[6-7].*|classifier.classifier.*)',
            # keys_to_freeze_regex = r'(.*resnet.*)',
            prefix_map={'classifier.resnet':'feature_model.layers.2.resnet', 'classifier.classifier':'classifier.layer',}
        )

class ImagenetNoisyRetinaBlurWRandomScalesXResNet2x18WDeepGazeIIE(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.125
    ckp_pth = f'/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18/1/checkpoints/model_checkpoint.pt'
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
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 1000), num_classes=1000,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_feature_extraction=True,
                                            setup_classification=False)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                use_1d_gkernels=True)
        # fixp = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
        #                                             CommonModelParams(self.input_size,), 'deepgaze2')
        fixp = DeepGazeIIE.ModelParams(DeepGazeIIE)
        retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,
                                                                        CommonModelParams(self.input_size), rnoise_p, rblur_p, fixp,
                                                                        target_downsample_factor=16, apply_retina_before_fixation=True)
        rp = SequentialLayers.ModelParams(SequentialLayers, [IdentityLayer.get_params(), retinafixp, resnet_p], CommonModelParams(self.input_size, activation=nn.Identity))
        cp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.input_size, 1000, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, cp)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=True
                )
            ),
            # SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            # OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=764, pct_start=0.1, anneal_strategy='linear'),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-4),
            ReduceLROnPlateauConfig(patience=3),
            logdir=LOGDIR, batch_size=64,
            seed_model_path=self.ckp_pth,
            # # keys_to_skip_regex = r'(.*resnet.[6-7].*|classifier.classifier.*)',
            # keys_to_freeze_regex = r'(.*resnet.*)',
            prefix_map={'classifier.resnet':'feature_model.layers.2.resnet', 'classifier.classifier':'classifier.layer',}
        )

class ImagenetNoisyRetinaBlurWRandomScalesXResNet2x18WDeepGazeII(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
    widen_stem = False
    noise_std = 0.125
    ckp_pth = f'/share/workhorse3/mshah1/biologically_inspired_models/iclr22_logs/imagenet_folder-0.0/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18/1/checkpoints/model_checkpoint.pt'
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
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 1000), num_classes=1000,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor, setup_feature_extraction=True,
                                            setup_classification=False)
        rblur_p = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform',
                                                use_1d_gkernels=True)
        # fixp = FixationPredictionNetwork.ModelParams(FixationPredictionNetwork,
        #                                             CommonModelParams(self.input_size,), 'deepgaze2')
        fixp = DeepGazeII.ModelParams(DeepGazeII)
        retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,
                                                                        CommonModelParams(self.input_size), rnoise_p, rblur_p, fixp,
                                                                        target_downsample_factor=16, apply_retina_before_fixation=True)
        rp = SequentialLayers.ModelParams(SequentialLayers, [IdentityLayer.get_params(), retinafixp, resnet_p], CommonModelParams(self.input_size, activation=nn.Identity))
        cp = LinearLayer.ModelParams(LinearLayer, CommonModelParams(self.input_size, 1000, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, rp, cp)
        return p

    def get_experiment_params(self) -> BaseExperimentConfig:
        nepochs = 60
        return TransferLearningExperimentConfig(
            LightningAdversarialTrainer.TrainerParams(LightningAdversarialTrainer,
                TrainingParams(logdir=LOGDIR, nepochs=nepochs, early_stop_patience=50, tracked_metric='val_accuracy',
                    tracking_mode='max', scheduler_step_after_epoch=True
                )
            ),
            # SGDOptimizerConfig(lr=0.2, weight_decay=5e-4, momentum=0.9, nesterov=True),
            # OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=764, pct_start=0.1, anneal_strategy='linear'),
            AdamOptimizerConfig(lr=0.001, weight_decay=5e-4),
            ReduceLROnPlateauConfig(patience=3),
            logdir=LOGDIR, batch_size=64,
            seed_model_path=self.ckp_pth,
            # # keys_to_skip_regex = r'(.*resnet.[6-7].*|classifier.classifier.*)',
            # keys_to_freeze_regex = r'(.*resnet.*)',
            prefix_map={'classifier.resnet':'feature_model.layers.2.resnet', 'classifier.classifier':'classifier.layer',}
        )