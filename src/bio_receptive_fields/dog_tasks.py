import torchvision
from adversarialML.biologically_inspired_models.src.models import (
    CommonModelParams, GeneralClassifier, SequentialLayers, XResNet18, ConvEncoder, ConvParams)
from adversarialML.biologically_inspired_models.src.DoG_kernels.models import DoGFilterbank, LearnableDoGFilterbank
from adversarialML.biologically_inspired_models.src.task_utils import *
from adversarialML.biologically_inspired_models.src.trainers import \
    LightningAdversarialTrainer
from mllib.optimizers.configs import OneCycleLRConfig, SGDOptimizerConfig
from mllib.runners.configs import BaseExperimentConfig, TrainingParams
from mllib.tasks.base_tasks import AbstractTask
from torch import nn

from adversarialML.biologically_inspired_models.src.locally_connected_layer import LocallyConnectedLayer

class Ecoset10RandColorDoGFBx1CyclicLRRandAugmentXResNet2x18(AbstractTask):
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
        dogfbp = DoGFilterbank.ModelParams(DoGFilterbank, 27, stride=4, padding=13)
        convp = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([1024, self.imgs_size, self.imgs_size], num_units=[64]),
                                        ConvParams([1], [1], [0]))
        dogfbp = SequentialLayers.ModelParams(SequentialLayers, [dogfbp, convp], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([64, self.imgs_size, self.imgs_size], 100), num_classes=100,
                                            widen_factor=self.widen_factor, stem_sizes=(64,64,64), drop_layers=[0,3])
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, dogfbp, resnet_p)
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

class Ecoset10SingleOpponentDoGFBx1CyclicLRRandAugmentXResNet2x18(AbstractTask):
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
        dogfbp = DoGFilterbank.ModelParams(DoGFilterbank, 27, stride=4, padding=13, application_mode='single_opponent')
        convp = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([3*1024, self.imgs_size, self.imgs_size], num_units=[64]),
                                        ConvParams([1], [1], [0]))
        dogfbp = SequentialLayers.ModelParams(SequentialLayers, [dogfbp, convp], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([64, self.imgs_size, self.imgs_size], 100), num_classes=100,
                                            widen_factor=self.widen_factor, stem_sizes=(64,64,64), drop_layers=[0,3])
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, dogfbp, resnet_p)
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

class Ecoset10GreyscaleDoGFBx1CyclicLRRandAugmentXResNet2x18(AbstractTask):
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
        dogfbp = DoGFilterbank.ModelParams(DoGFilterbank, 27, stride=4, padding=13, application_mode='greyscale')
        convp = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([1024, self.imgs_size, self.imgs_size], num_units=[64]),
                                        ConvParams([1], [1], [0]))
        dogfbp = SequentialLayers.ModelParams(SequentialLayers, [dogfbp, convp], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([64, self.imgs_size, self.imgs_size], 100), num_classes=100,
                                            widen_factor=self.widen_factor, stem_sizes=(64,64,64), drop_layers=[0,3])
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, dogfbp, resnet_p)
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
    
class Ecoset100RandColorDoGFBx1CyclicLRRandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
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
        # # Pointing to a folder with only the test set, and some dummy train and val data. 
        # # Use this on workhorse to avoid delay due to slow NFS.
        # p.datafolder = f'{logdir_root}/ecoset-100/eval_dataset_dir'
        return p

    def get_model_params(self):
        dogfbp = DoGFilterbank.ModelParams(DoGFilterbank, 27, stride=4, padding=13)
        convp = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([1024, self.imgs_size, self.imgs_size], num_units=[64]),
                                        ConvParams([1], [1], [0]))
        dogfbp = SequentialLayers.ModelParams(SequentialLayers, [dogfbp, convp], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([64, self.imgs_size, self.imgs_size], 100), num_classes=100,
                                            widen_factor=self.widen_factor, stem_sizes=(64,64,64), drop_layers=[0,3])
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, dogfbp, resnet_p)
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
            logdir=LOGDIR, batch_size=128
        )
    
class Ecoset100RandColorDoGFBx2CyclicLRRandAugmentXResNet2x18(Ecoset100RandColorDoGFBx1CyclicLRRandAugmentXResNet2x18):
    def get_model_params(self):
        dogfbp1 = DoGFilterbank.ModelParams(DoGFilterbank, 11, stride=2, padding=5)
        convp1 = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([512, self.imgs_size, self.imgs_size], num_units=[64]),
                                        ConvParams([1], [1], [0]))
        dogfbp2 = DoGFilterbank.ModelParams(DoGFilterbank, 7, in_channels=64, stride=2, padding=3,
                                            application_mode='all_channels', normalize_input=False)
        convp2 = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([512, self.imgs_size, self.imgs_size], num_units=[64]),
                                        ConvParams([1], [1], [0]))
        dogfbp = SequentialLayers.ModelParams(SequentialLayers, [dogfbp1, convp1, dogfbp2, convp2], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([64, self.imgs_size, self.imgs_size], 100), num_classes=100,
                                            widen_factor=self.widen_factor, stem_sizes=(64,64,64), drop_layers=[0,3])
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, dogfbp, resnet_p)
        return p
    
# class Ecoset100SingleOpponentDoGFBx1CyclicLRRandAugmentXResNet2x18(Ecoset100RandColorDoGFBx1CyclicLRRandAugmentXResNet2x18):
#     def get_model_params(self):
#         dogfbp = DoGFilterbank.ModelParams(DoGFilterbank, 27, stride=4, padding=13, application_mode='single_opponent')
#         convp = ConvEncoder.ModelParams(ConvEncoder,
#                                         CommonModelParams([1024, self.imgs_size, self.imgs_size], num_units=[64]),
#                                         ConvParams([1], [1], [0]))
#         dogfbp = SequentialLayers.ModelParams(SequentialLayers, [dogfbp, convp], CommonModelParams(self.input_size, activation=nn.Identity))
#         resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([64, self.imgs_size, self.imgs_size], 100), num_classes=100,
#                                             widen_factor=self.widen_factor, stem_sizes=(64,64,64), drop_layers=[0,3])
#         p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, dogfbp, resnet_p)
#         return p

class Ecoset100LearnableDoGFBx1CyclicLRRandAugmentXResNet2x18(AbstractTask):
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
    widen_factor = 2
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
        # # Pointing to a folder with only the test set, and some dummy train and val data. 
        # # Use this on workhorse to avoid delay due to slow NFS.
        # p.datafolder = f'{logdir_root}/ecoset-100/eval_dataset_dir'
        return p

    def get_model_params(self):
        dogfbp = LearnableDoGFilterbank.ModelParams(LearnableDoGFilterbank, 1024, 27, stride=4, padding=13)
        convp = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([1024, self.imgs_size, self.imgs_size], num_units=[64]),
                                        ConvParams([1], [1], [0]))
        dogfbp = SequentialLayers.ModelParams(SequentialLayers, [dogfbp, convp], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([64, self.imgs_size, self.imgs_size], 100), num_classes=100,
                                            widen_factor=self.widen_factor, stem_sizes=(64,64,64), drop_layers=[0,3])
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, dogfbp, resnet_p)
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
            logdir=LOGDIR, batch_size=128)