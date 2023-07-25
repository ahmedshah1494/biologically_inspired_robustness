import numpy as np
import torchvision
from adversarialML.biologically_inspired_models.src.models import (
    CommonModelParams, GeneralClassifier, SequentialLayers, XResNet18, ConvEncoder, ConvParams, IdentityLayer)
from adversarialML.biologically_inspired_models.src.bio_receptive_fields.dog_filterbank import DoGFilterbank, LearnableDoGFilterbank, RandomFilterbank
from adversarialML.biologically_inspired_models.src.task_utils import *
from adversarialML.biologically_inspired_models.src.trainers import \
    LightningAdversarialTrainer
from mllib.optimizers.configs import OneCycleLRConfig, SGDOptimizerConfig
from mllib.runners.configs import BaseExperimentConfig, TrainingParams
from mllib.tasks.base_tasks import AbstractTask
from torch import nn

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
            logdir=LOGDIR, batch_size=32
        )
    
class Ecoset10RandColorDoGFBx2CyclicLRRandAugmentXResNet2x18(Ecoset10RandColorDoGFBx1CyclicLRRandAugmentXResNet2x18):
    def get_model_params(self):
        dogfbp1 = DoGFilterbank.ModelParams(DoGFilterbank, 11, stride=2, padding=5, num_scales=4, scaling_factor=np.sqrt(2))
        convp1 = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([512, self.imgs_size, self.imgs_size], num_units=[32]),
                                        ConvParams([1], [1], [0]))
        dogfbp2 = DoGFilterbank.ModelParams(DoGFilterbank, 7, in_channels=32, stride=2, padding=3, num_scales=2, num_shapes=4, num_orientations=4,
                                            application_mode='all_channels_replicated', normalize_input=False)
        convp2 = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([2048, self.imgs_size, self.imgs_size], num_units=[64]),
                                        ConvParams([1], [1], [0]))
        dogfbp = SequentialLayers.ModelParams(SequentialLayers, [dogfbp1, convp1, dogfbp2, convp2], CommonModelParams(self.input_size, activation=nn.Identity))
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
            logdir=LOGDIR, batch_size=32
        )

class Ecoset10RandConvx1CyclicLRRandAugmentXResNet2x18(AbstractTask):
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
        dogfbp = RandomFilterbank.ModelParams(RandomFilterbank, 27, 3, 1024, stride=4, padding=13)
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
            logdir=LOGDIR, batch_size=64
        )

class Ecoset10RandConvx2CyclicLRRandAugmentXResNet2x18(Ecoset10RandConvx1CyclicLRRandAugmentXResNet2x18):
    def get_model_params(self):
        dogfbp1 = RandomFilterbank.ModelParams(RandomFilterbank, 11, 3, 512, stride=2, padding=5)
        convp1 = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([512, self.imgs_size, self.imgs_size], num_units=[32]),
                                        ConvParams([1], [1], [0]))
        dogfbp2 = RandomFilterbank.ModelParams(RandomFilterbank, 7, 32, 2048, stride=2, padding=3, normalize_input=False)
        convp2 = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([2048, self.imgs_size, self.imgs_size], num_units=[64]),
                                        ConvParams([1], [1], [0]))
        dogfbp = SequentialLayers.ModelParams(SequentialLayers, [dogfbp1, convp1, dogfbp2, convp2], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([64, self.imgs_size, self.imgs_size], 100), num_classes=100,
                                            widen_factor=self.widen_factor, stem_sizes=(64,64,64), drop_layers=[0,3])
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, dogfbp, resnet_p)
        return p
    
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
    
class Ecoset100Pruned512DoGFBx1CyclicLRRandAugmentXResNet2x18(Ecoset100RandColorDoGFBx1CyclicLRRandAugmentXResNet2x18):
    def get_model_params(self):
        dogfbp = DoGFilterbank.ModelParams(DoGFilterbank, 27, padding=13, stride=2,
                                           kernel_path='/ocean/projects/cis220031p/mshah1/biologically_inspired_models/bioRF_logs//DoG_filterbanks/Ecoset100AllColorsDoGFBPruningTask/train/512_filterbank_kernels.pt')
        convp = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([512, self.imgs_size, self.imgs_size], num_units=[64]),
                                        ConvParams([1], [1], [0]))
        dogfbp = SequentialLayers.ModelParams(SequentialLayers, [dogfbp, convp], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([64, self.imgs_size, self.imgs_size], 100), num_classes=100,
                                            widen_factor=self.widen_factor, stem_sizes=(64,64,64), drop_layers=[0])
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
            logdir=LOGDIR, batch_size=64
        )
    
class Ecoset100RandColorDoGFBx2CyclicLRRandAugmentXResNet2x18(Ecoset100RandColorDoGFBx1CyclicLRRandAugmentXResNet2x18):
    def get_model_params(self):
        dogfbp1 = DoGFilterbank.ModelParams(DoGFilterbank, 11, stride=2, padding=5, num_scales=4, scaling_factor=np.sqrt(2))
        convp1 = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([512, self.imgs_size, self.imgs_size], num_units=[32]),
                                        ConvParams([1], [1], [0]))
        dogfbp2 = DoGFilterbank.ModelParams(DoGFilterbank, 7, in_channels=32, stride=2, padding=3, num_scales=2, num_shapes=4, num_orientations=4,
                                            application_mode='all_channels_replicated', normalize_input=False)
        convp2 = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([2048, self.imgs_size, self.imgs_size], num_units=[64]),
                                        ConvParams([1], [1], [0]))
        dogfbp = SequentialLayers.ModelParams(SequentialLayers, [dogfbp1, convp1, dogfbp2, convp2], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([64, self.imgs_size, self.imgs_size], 100), num_classes=100,
                                            widen_factor=self.widen_factor, stem_sizes=(64,64,64), drop_layers=[0,3])
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, dogfbp, resnet_p)
        return p
    
class Ecoset100AllColorsDoGFBPruningTask(AbstractTask):
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
        dogfbp = DoGFilterbank.ModelParams(DoGFilterbank, 27, padding=13, application_mode='all_channels_replicated')
        return dogfbp
    
class ImagenetRandColorDoGFBx2CyclicLRRandAugmentXResNet2x18(AbstractTask):
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
        dogfbp1 = DoGFilterbank.ModelParams(DoGFilterbank, 11, stride=2, padding=5, num_scales=4, scaling_factor=np.sqrt(2))
        convp1 = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([512, self.imgs_size, self.imgs_size], num_units=[64]),
                                        ConvParams([1], [1], [0]))
        dogfbp2 = DoGFilterbank.ModelParams(DoGFilterbank, 7, in_channels=64, stride=2, padding=3, num_scales=2, num_shapes=4, num_orientations=4,
                                            application_mode='all_channels_replicated', normalize_input=False)
        convp2 = ConvEncoder.ModelParams(ConvEncoder,
                                        CommonModelParams([2048, self.imgs_size, self.imgs_size], num_units=[64]),
                                        ConvParams([1], [1], [0]))
        dogfbp = SequentialLayers.ModelParams(SequentialLayers, [dogfbp1, convp1, dogfbp2, convp2], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([64, self.imgs_size, self.imgs_size], 1000), num_classes=1000,
                                            widen_factor=self.widen_factor, stem_sizes=(64,64,64), drop_layers=[0,3])
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, dogfbp, resnet_p)
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

