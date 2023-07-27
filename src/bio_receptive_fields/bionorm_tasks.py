import torchvision
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import NormalizationLayer
from adversarialML.biologically_inspired_models.src.models import (
    CommonModelParams, XResNet18)
from adversarialML.biologically_inspired_models.src.trainers import LightningAdversarialTrainer
from mllib.optimizers.configs import (OneCycleLRConfig, SGDOptimizerConfig)
from mllib.runners.configs import BaseExperimentConfig, TrainingParams
from mllib.tasks.base_tasks import AbstractTask
from adversarialML.biologically_inspired_models.src.task_utils import *
from adversarialML.biologically_inspired_models.src.bio_receptive_fields.bionorm import BioNormWrapper, BioNorm

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
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 1000), num_classes=1000,
                                            normalization_layer_params=NormalizationLayer.get_params(),
                                            widen_factor=self.widen_factor)
        p = BioNormWrapper.ModelParams(BioNormWrapper, resnet_p, 5)
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

class Ecoset10InputBioNormCyclicLRRandAugmentBioNormXResNet2x18(AbstractTask):
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
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 10), num_classes=10,
                                            normalization_layer_params=BioNorm.ModelParams(BioNorm, 3, 7, affine=False,
                                                                                           center=True, channel_pooling=False,
                                                                                           pad_before_sum=True),
                                            widen_factor=self.widen_factor)
        p = BioNormWrapper.ModelParams(BioNormWrapper, resnet_p,
                                       BioNorm.ModelParams(BioNorm, kernel_size=5, center=True))
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

class ImagenetCyclicLRRandAugmentBioNormXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
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
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams(self.input_size, 1000), num_classes=1000,
                                            normalization_layer_params=BioNorm.ModelParams(BioNorm, 3, 3, affine=False, center=True, pad=True),
                                            widen_factor=self.widen_factor)
        p = BioNormWrapper.ModelParams(BioNormWrapper, resnet_p, 5)
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