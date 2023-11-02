import numpy as np
import torchvision
from adversarialML.biologically_inspired_models.src.models import (
    CommonModelParams, GeneralClassifier, SequentialLayers, XResNet18, ConvEncoder, ConvParams, IdentityLayer, ActivationLayer)
from adversarialML.biologically_inspired_models.src.bio_receptive_fields.torchdeepretina import TorchDeepRetina
from adversarialML.biologically_inspired_models.src.task_utils import *
from adversarialML.biologically_inspired_models.src.trainers import \
    LightningAdversarialTrainer
from mllib.optimizers.configs import OneCycleLRConfig, SGDOptimizerConfig
from mllib.runners.configs import BaseExperimentConfig, TrainingParams
from mllib.tasks.base_tasks import AbstractTask
from torch import nn

class Ecoset10TDRetinaCyclicLRRandAugmentXResNet2x18(AbstractTask):
    imgs_size = 50
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
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
        tdrp = TorchDeepRetina.ModelParams(TorchDeepRetina, '/home/mshah1/projects/adversarialML/biologically_inspired_models/torch-deep-retina/models/15-11-21a_naturalscene.pt',
                                           upscale_factor=224/self.imgs_size)
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([8, self.imgs_size, self.imgs_size], 10), num_classes=10,
                                            widen_factor=self.widen_factor, drop_layers=[0])
        convp1 = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([8, self.imgs_size, self.imgs_size], num_units=[32]),
                                        ConvParams([1], [1], [0]))
        tdrp = SequentialLayers.ModelParams(SequentialLayers, [tdrp, convp1], CommonModelParams(self.input_size, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, tdrp, resnet_p)
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
    
class Ecoset10TDRetinaIS224CyclicLRRandAugmentXResNet2x18(Ecoset10TDRetinaCyclicLRRandAugmentXResNet2x18):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]

    def get_model_params(self):
        tdrp = TorchDeepRetina.ModelParams(TorchDeepRetina, '/home/mshah1/projects/adversarialML/biologically_inspired_models/torch-deep-retina/models/15-11-21a_naturalscene.pt',
                                           upscale_factor=224/self.imgs_size)
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([8, self.imgs_size, self.imgs_size], 10), num_classes=10,
                                            widen_factor=self.widen_factor)
        convp1 = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([8, self.imgs_size, self.imgs_size], num_units=[32]),
                                        ConvParams([1], [1], [0]))
        tdrp = SequentialLayers.ModelParams(SequentialLayers, [tdrp, convp1], CommonModelParams(self.input_size, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, tdrp, resnet_p)
        return p

class Ecoset100TDRetinaCyclicLRRandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
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
        tdrp = TorchDeepRetina.ModelParams(TorchDeepRetina, '/home/mshah1/projects/adversarialML/biologically_inspired_models/torch-deep-retina/models/15-11-21a_naturalscene.pt',
                                           upscale_factor=224/self.imgs_size)
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([8, self.imgs_size, self.imgs_size], 100), num_classes=100,
                                            widen_factor=self.widen_factor)
        convp1 = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([8, self.imgs_size, self.imgs_size], num_units=[32]),
                                        ConvParams([1], [1], [0]))
        tdrp = SequentialLayers.ModelParams(SequentialLayers, [tdrp, convp1], CommonModelParams(self.input_size, activation=nn.Identity))
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, tdrp, resnet_p)
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
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=1839, pct_start=0.2, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64
        )