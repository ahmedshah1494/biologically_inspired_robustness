import numpy as np
import torchvision
from adversarialML.biologically_inspired_models.src.models import (
    CommonModelParams, GeneralClassifier, SequentialLayers, XResNet18, ConvEncoder, ConvParams)
from adversarialML.biologically_inspired_models.src.bio_receptive_fields.dog_filterbank import DoGFilterbank
from adversarialML.biologically_inspired_models.src.bio_receptive_fields.local_contrast import LocalContrast, ContrastNormalize
from adversarialML.biologically_inspired_models.src.task_utils import *
from adversarialML.biologically_inspired_models.src.trainers import \
    LightningAdversarialTrainer
from mllib.optimizers.configs import OneCycleLRConfig, SGDOptimizerConfig
from mllib.runners.configs import BaseExperimentConfig, TrainingParams
from mllib.tasks.base_tasks import AbstractTask
from torch import nn

class Ecoset10CyclicLRRandAugmentBioNormXResNet2x18(AbstractTask):
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
        lcp = LocalContrast.ModelParams(LocalContrast, 5, 3)
        dogfbp = DoGFilterbank.ModelParams(DoGFilterbank, 11, padding=5, num_scales=4, stride=2, num_orientations=4, base_std=0.32)
        cnp = ContrastNormalize.ModelParams(ContrastNormalize, lcp, dogfbp, 15)
        convp = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([256*3, self.imgs_size, self.imgs_size], num_units=[64]),
                                        ConvParams([1], [1], [0]))
        fp = SequentialLayers.ModelParams(SequentialLayers, [cnp, convp], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([64, self.imgs_size, self.imgs_size], 100), num_classes=100,
                                            widen_factor=self.widen_factor, stem_sizes=(64,64,64), drop_layers=[0,])
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, fp, resnet_p)
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
            OneCycleLRConfig(max_lr=0.4, epochs=nepochs, steps_per_epoch=376, pct_start=0.2, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=32
        )