import torchvision
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import NormalizationLayer
from adversarialML.biologically_inspired_models.src.models import (
    CommonModelParams, GeneralClassifier, SequentialLayers, XResNet34, XResNet18, XResNet50, WideResnet, CORnetS,
    ActivationLayer, BatchNorm2DLayer, LogitAverageEnsembler, SupervisedContrastiveTrainingWrapper, IdentityLayer,
    XResNetClassifierWithReconstructionLoss, XResNetClassifierWithEnhancer, XResNetClassifierWithDeepResidualEnhancer,
    FixationPredictionNetwork, ConvEncoder, ConvParams, FixationHeatmapPredictionNetwork, RetinaFilterFixationPrediction)
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import LinearLayer
from adversarialML.biologically_inspired_models.src.retina_blur2 import RetinaBlurFilter as RBlur2
from adversarialML.biologically_inspired_models.src.retina_preproc import (
    RetinaBlurFilter, RetinaNonUniformPatchEmbedding, RetinaWarp, GaussianNoiseLayer)
from adversarialML.biologically_inspired_models.src.supconloss import \
    TwoCropTransform
from adversarialML.biologically_inspired_models.src.trainers import LightningAdversarialTrainer, FixationPointLightningAdversarialTrainer,ClickmeImportanceMapLightningAdversarialTrainer
from mllib.optimizers.configs import (AdamOptimizerConfig,
                                      CosineAnnealingWarmRestartsConfig,
                                      CyclicLRConfig, LinearLRConfig,
                                      ReduceLROnPlateauConfig,
                                      SequentialLRConfig, OneCycleLRConfig, SGDOptimizerConfig)
from mllib.runners.configs import BaseExperimentConfig, TrainingParams
from mllib.tasks.base_tasks import AbstractTask
from torch import nn
from adversarialML.biologically_inspired_models.src.task_utils import *
from mllib.datasets.dataset_factory import SupportedDatasets
from adversarialML.biologically_inspired_models.src.runners import TransferLearningExperimentConfig

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
        p = get_ecoset100folder_params(num_train=400_000, num_test=5000, train_transforms=[
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
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=1563, pct_start=0.05, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64
        )

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
            ],
        test_transforms=[
            torchvision.transforms.Resize(self.imgs_size),
            torchvision.transforms.CenterCrop(self.imgs_size),
        ])
        return p
    
    def get_model_params(self):
        rnoise_p = GaussianNoiseLayer.ModelParams(GaussianNoiseLayer, max_input_size=self.input_size, std=self.noise_std)
        rblur_p = RBlur2.ModelParams(RBlur2, self.input_size, batch_size=32, cone_std=0.12, 
                                                rod_std=0.09, max_rod_density=0.12, view_scale='random_uniform', loc_mode='center',
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
            AdamOptimizerConfig(lr=0.2, weight_decay=5e-4),
            OneCycleLRConfig(max_lr=0.001, epochs=nepochs, steps_per_epoch=141, pct_start=0.1, anneal_strategy='linear'),
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
        retinafixp = RetinaFilterFixationPrediction.ModelParams(RetinaFilterFixationPrediction,CommonModelParams(self.input_size), rnoise_p, rblur_p, fixp,
                                                        fixation_model_ckp='/share/workhorse3/mshah1/biologically_inspired_models/logs/clickme-0.0/ClickmeNoisyRetinaBlur2S2500WRandomScalesFixationHeatmapPredictor/2/checkpoints/model_checkpoint.pt'
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