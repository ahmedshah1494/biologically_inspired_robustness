import numpy as np
import torchvision
from adversarialML.biologically_inspired_models.src.models import (
    CommonModelParams, GeneralClassifier, SequentialLayers, XResNet18, ConvEncoder, ConvParams, IdentityLayer, ActivationLayer)
from adversarialML.biologically_inspired_models.src.bio_receptive_fields.color_opponent_rf import LinearColorOpponentRF
from adversarialML.biologically_inspired_models.src.bio_receptive_fields.bionorm import BioNorm, BioNormWrapper
from adversarialML.biologically_inspired_models.src.retina_preproc import NeuronalGaussianNoiseLayer, GaussianNoiseLayer
from adversarialML.biologically_inspired_models.src.task_utils import *
from adversarialML.biologically_inspired_models.src.trainers import \
    LightningAdversarialTrainer
from mllib.optimizers.configs import OneCycleLRConfig, SGDOptimizerConfig
from mllib.runners.configs import BaseExperimentConfig, TrainingParams
from mllib.tasks.base_tasks import AbstractTask
from torch import nn

class Ecoset10LCOFBx1CyclicLRRandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
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
        return p

    def get_model_params(self):
        lcofbp = LinearColorOpponentRF.ModelParams(LinearColorOpponentRF, 1024, 27, stride=4, padding=13)
        convp = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([1024, self.imgs_size, self.imgs_size], num_units=[64]),
                                        ConvParams([1], [1], [0]))
        lcofbp = SequentialLayers.ModelParams(SequentialLayers, [lcofbp, convp], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([64, self.imgs_size, self.imgs_size], 10), num_classes=10,
                                            widen_factor=self.widen_factor, stem_sizes=(64,64,64), drop_layers=[0,3])
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, lcofbp, resnet_p)
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

class Ecoset10NoisyLCOFBx1CyclicLRRandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
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
        return p

    def get_model_params(self):
        noisep1 = NeuronalGaussianNoiseLayer.ModelParams(NeuronalGaussianNoiseLayer, std=0.35, mean=0.07,
                                                         add_deterministic_noise_during_inference=True,
                                                         max_input_size=[1024,56,56])
        relup = ActivationLayer.ModelParams(ActivationLayer, nn.ReLU)
        lcofbp = LinearColorOpponentRF.ModelParams(LinearColorOpponentRF, 1024, 27, stride=4, padding=13)
        convp = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([1024, self.imgs_size, self.imgs_size], num_units=[64]),
                                        ConvParams([1], [1], [0]))
        lcofbp = SequentialLayers.ModelParams(SequentialLayers, [lcofbp, noisep1, convp], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([64, self.imgs_size, self.imgs_size], 10), num_classes=10,
                                            widen_factor=self.widen_factor, stem_sizes=(64,64,64), drop_layers=[0,3])
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, lcofbp, resnet_p)
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