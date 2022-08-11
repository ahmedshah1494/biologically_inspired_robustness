from copy import deepcopy
from typing import List, Tuple
from mllib.tasks.base_tasks import AbstractTask
from mllib.models.base_models import MLP
from mllib.runners.configs import BaseExperimentConfig
from mllib.optimizers.configs import SGDOptimizerConfig, ReduceLROnPlateauConfig, AdamOptimizerConfig, CosineAnnealingWarmRestartsConfig
from mllib.adversarial.attacks import AttackParamFactory, SupportedAttacks, SupportedBackend

from adversarialML.biologically_inspired_models.src.models import ConvEncoder
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import MLPMixer, MixerBlock
from mlp_mixer_models import LinearLayer, MixerMLP
from runners import AdversarialExperimentConfig

from tasks import get_cifar10_params, set_SGD_params, set_adv_params, set_common_training_params
from torch import nn
import torchvision

class Cifar10AutoAugmentMLPMixerTask(AbstractTask):
    hidden_size = 128
    input_size = [3, 32, 32]
    patch_size = 4
    num_blocks = 4
    mlpc_hidden = 512
    mlps_hidden = 64

    def get_dataset_params(self):
        p = get_cifar10_params(num_train=50_000)
        p.custom_transforms = (
            torchvision.transforms.Compose([
                torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.CIFAR10),
                torchvision.transforms.ToTensor()
            ]),
            torchvision.transforms.ToTensor()
        )
        return p
    
    def get_experiment_params(self):
        p = AdversarialExperimentConfig()
        set_common_training_params(p)
        # set_SGD_params(p)
        # p.optimizer_config.lr = 0.05
        p.optimizer_config = AdamOptimizerConfig()
        p.optimizer_config.weight_decay = 5e-5
        p.scheduler_config = CosineAnnealingWarmRestartsConfig()
        p.training_params.early_stop_patience = 10
        p.scheduler_config.T_0 = p.training_params.nepochs
        p.scheduler_config.eta_min = 1e-6
        p.batch_size = 128
        test_eps = [0.0, 0.008, 0.016, 0.024, 0.032, 0.048, 0.064]
        set_adv_params(p, test_eps)
        dsp = self.get_dataset_params()
        p.exp_name = f'{dsp.max_num_train//1000}K'
        return p

    def get_model_params(self):
        patch_params: ConvEncoder.ModelParams = ConvEncoder.get_params()
        patch_params.common_params.input_size = self.input_size
        patch_params.common_params.num_units = [self.hidden_size]
        patch_params.conv_params.kernel_sizes = [self.patch_size]
        patch_params.conv_params.padding = [0]
        patch_params.conv_params.strides = [self.patch_size]

        n_patches = (self.input_size[1] // self.patch_size)**2

        mlpc_params: MixerMLP.ModelParams = MixerMLP.get_params()
        mlpc_params.common_params.activation = nn.GELU
        mlpc_params.common_params.dropout_p = 0.
        mlpc_params.common_params.input_size = [self.hidden_size]
        mlpc_params.common_params.num_units = self.mlpc_hidden

        mlps_params: MixerMLP.ModelParams = MixerMLP.get_params()
        mlps_params.common_params.activation = nn.GELU
        mlps_params.common_params.dropout_p = 0.
        mlps_params.common_params.input_size = [n_patches]
        mlps_params.common_params.num_units = self.mlps_hidden

        cls_params: LinearLayer.ModelParams = LinearLayer.get_params()
        cls_params.common_params.input_size = self.hidden_size
        cls_params.common_params.num_units = 10
        cls_params.common_params.activation = nn.Identity

        block_params: MixerBlock.ModelParams = MixerBlock.get_params()
        block_params.channel_mlp_params = mlpc_params
        block_params.spatial_mlp_params = mlps_params
        block_params.common_params.input_size = [n_patches, self.hidden_size]

        mixer_params: MLPMixer.ModelParams = MLPMixer.get_params()
        mixer_params.common_params.input_size = self.input_size
        mixer_params.patch_gen_params = patch_params
        mixer_params.mixer_block_params = [deepcopy(block_params) for i in range(self.num_blocks)]
        mixer_params.classifier_params = cls_params
        return mixer_params

class Cifar10AutoAugmentMLPMixer8LTask(Cifar10AutoAugmentMLPMixerTask):
    num_blocks = 8
        