from adversarialML.biologically_inspired_models.src.mlp_mixer_models import NormalizationLayer
from adversarialML.biologically_inspired_models.src.models import PretrainedTimmModel
from adversarialML.biologically_inspired_models.src.task_utils import *
from mllib.tasks.base_tasks import AbstractTask
from adversarialML.biologically_inspired_models.src.trainers import MixedPrecisionAdversarialTrainer, LightningAdversarialTrainer
from mllib.optimizers.configs import (AdamOptimizerConfig,
                                      CosineAnnealingWarmRestartsConfig,
                                      CyclicLRConfig, LinearLRConfig,
                                      ReduceLROnPlateauConfig,
                                      SequentialLRConfig, OneCycleLRConfig, SGDOptimizerConfig)
from mllib.runners.configs import BaseExperimentConfig, TrainingParams
import timm

class BaseImagenetEvalTask(AbstractTask):
    def get_dataset_params(self) :
        p = get_imagenet_folder_params(num_train=1_275_000,
            test_transforms=[
                torchvision.transforms.Resize(self.imgs_size),
                torchvision.transforms.CenterCrop(self.imgs_size),
            ])
        # Pointing to a folder with only the test set, and some dummy train and val data. 
        # Use this on workhorse to avoid delay due to slow NFS.
        p.datafolder = f'{logdir_root}/imagenet/eval_dataset_dir'
        return p
    
    def get_model_params(self):
        p = PretrainedTimmModel.get_params()
        p.model_name = self.model_name
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
            OneCycleLRConfig(max_lr=0.2, epochs=nepochs, steps_per_epoch=4916, pct_start=0.2, anneal_strategy='linear'),
            logdir=LOGDIR, batch_size=64 # 4GPUS 256 batch
        )

class ImagenetPretrainedTimmEvaGEvalTask(BaseImagenetEvalTask):
    imgs_size = 336
    input_size = [3, imgs_size, imgs_size]
    model_name = 'eva_giant_patch14_336.m30m_ft_in22k_in1k'

class ImagenetPretrainedTimmEfficientNetL2EvalTask(BaseImagenetEvalTask):
    model_name = 'tf_efficientnet_l2.ns_jft_in1k_475'
    imgs_size = 475
    input_size = [3, imgs_size, imgs_size]

class ImagenetPretrainedTimmBeitLEvalTask(BaseImagenetEvalTask):
    model_name = 'beit_large_patch16_224.in22k_ft_in22k_in1k'
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]


class ImagenetPretrainedTimmMaxViTXLEvalTask(BaseImagenetEvalTask):
    model_name = 'maxvit_xlarge_tf_384.in21k_ft_in1k'
    imgs_size = 384
    input_size = [3, imgs_size, imgs_size]


class ImagenetPretrainedTimmViTHEvalTask(BaseImagenetEvalTask):
    model_name = 'vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k'
    imgs_size = 224
    input_size = [3, imgs_size, imgs_size]
