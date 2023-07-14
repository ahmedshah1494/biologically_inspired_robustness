import torchvision
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import \
    NormalizationLayer
from adversarialML.biologically_inspired_models.src.models import (
    CommonModelParams, GeneralClassifier, SequentialLayers, XResNet18)
from adversarialML.biologically_inspired_models.src.retina_preproc import \
    VOneBlock
from adversarialML.biologically_inspired_models.src.task_utils import *
from adversarialML.biologically_inspired_models.src.trainers import \
    LightningAdversarialTrainer
from mllib.optimizers.configs import OneCycleLRConfig, SGDOptimizerConfig
from mllib.runners.configs import BaseExperimentConfig, TrainingParams
from mllib.tasks.base_tasks import AbstractTask
from torch import nn


class Ecoset100SingleOpponentVOneBlockCyclicLRRandAugmentXResNet2x18(AbstractTask):
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
        # Pointing to a folder with only the test set, and some dummy train and val data. 
        # Use this on workhorse to avoid delay due to slow NFS.
        # p.datafolder = f'{logdir_root}/ecoset-100/eval_dataset_dir'
        return p

    def get_model_params(self):
        vonep = VOneBlock.ModelParams(VOneBlock, add_deterministic_noise_during_inference=True, color_mode='exhaustive_single_opponent')
        normp = NormalizationLayer.ModelParams(NormalizationLayer, [0.5]*3, [0.5]*3)
        v1p = SequentialLayers.ModelParams(SequentialLayers, [normp, vonep], CommonModelParams(self.input_size, activation=nn.Identity))
        resnet_p = XResNet18.ModelParams(XResNet18, CommonModelParams([64, self.imgs_size, self.imgs_size], 100), num_classes=100,
                                            widen_factor=self.widen_factor, stem_sizes=(64,64,64), drop_layers=[0,3])
        p = GeneralClassifier.ModelParams(GeneralClassifier, self.input_size, v1p, resnet_p)
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
            OneCycleLRConfig(max_lr=0.1, epochs=nepochs, steps_per_epoch=1839*2, pct_start=0.2, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=32
        )