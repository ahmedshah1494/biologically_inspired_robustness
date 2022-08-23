from copy import deepcopy
from typing import List, Tuple
from mllib.tasks.base_tasks import AbstractTask
from mllib.models.base_models import MLP
from mllib.runners.configs import BaseExperimentConfig
from mllib.optimizers.configs import SGDOptimizerConfig, ReduceLROnPlateauConfig, AdamOptimizerConfig, CosineAnnealingWarmRestartsConfig, SequentialLRConfig, LinearLRConfig
from mllib.adversarial.attacks import AttackParamFactory, SupportedAttacks, SupportedBackend

from adversarialML.biologically_inspired_models.src.models import ConvEncoder, ConsistentActivationLayer, ScanningConsistentActivationLayer, SequentialLayers
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import MLPMixer, MixerBlock
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import ConsistentActivationMixerBlock, ConsistentActivationMixerMLP, FirstNExtractionClassifier, LinearLayer, MixerMLP
from adversarialML.biologically_inspired_models.src.runners import AdversarialExperimentConfig, ConsistentActivationAdversarialExperimentConfig
from adversarialML.biologically_inspired_models.src.mlp_mixer_models import UnfoldPatchExtractor
from adversarialML.biologically_inspired_models.src.retina_preproc import RetinaBlurFilter, RetinaSampleFilter, RetinaNonUniformPatchEmbedding
from mlp_mixer_models import NormalizationLayer

from tasks import get_cifar10_params, set_SGD_params, set_adv_params, set_common_training_params
from torch import nn
import torchvision

from adversarialML.biologically_inspired_models.src.trainers import ActivityOptimizationSchedule

class Cifar10AutoAugmentMLPMixerTask(AbstractTask):
    hidden_size = 128
    input_size = [3, 32, 32]
    patch_size = 4
    num_blocks = 4
    mlpc_hidden = 512
    mlps_hidden = 64

    def _get_n_patches(self):
        return (self.input_size[1] // self.patch_size)**2

    def get_dataset_params(self):
        p = get_cifar10_params(num_train=50_000)
        p.custom_transforms = (
            torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.CIFAR10),
                torchvision.transforms.ToTensor()
            ]),
            torchvision.transforms.ToTensor()
        )
        return p
    
    def get_experiment_params(self):
        p = AdversarialExperimentConfig()
        set_common_training_params(p)
        p.training_params.nepochs = 300
        p.optimizer_config = AdamOptimizerConfig()
        p.optimizer_config.weight_decay = 5e-5
        p.scheduler_config = CosineAnnealingWarmRestartsConfig()
        p.training_params.early_stop_patience = 100
        p.training_params.tracked_metric = 'val_accuracy'
        p.training_params.tracking_mode = 'max'
        p.scheduler_config.T_0 = p.training_params.nepochs
        p.scheduler_config.eta_min = 1e-6
        p.batch_size = 128
        test_eps = [0.0, 0.008, 0.016, 0.024, 0.032, 0.048, 0.064]
        set_adv_params(p, test_eps)
        dsp = self.get_dataset_params()
        p.exp_name = f'{dsp.max_num_train//1000}K'
        return p

    def _get_patch_params(self):
        patch_params: ConvEncoder.ModelParams = ConvEncoder.get_params()
        patch_params.common_params.input_size = self.input_size
        patch_params.common_params.num_units = [self.hidden_size]
        patch_params.common_params.activation = nn.Identity
        patch_params.conv_params.kernel_sizes = [self.patch_size]
        patch_params.conv_params.padding = [0]
        patch_params.conv_params.strides = [self.patch_size]
        return patch_params

    def _get_mlpc_params(self):
        mlpc_params: MixerMLP.ModelParams = MixerMLP.get_params()
        mlpc_params.common_params.activation = nn.GELU
        mlpc_params.common_params.dropout_p = 0.
        mlpc_params.common_params.input_size = [self.hidden_size]
        mlpc_params.common_params.num_units = self.mlpc_hidden
        return mlpc_params

    def _get_mlps_params(self):
        n_patches = self._get_n_patches()
        mlps_params: MixerMLP.ModelParams = MixerMLP.get_params()
        mlps_params.common_params.activation = nn.GELU
        mlps_params.common_params.dropout_p = 0.
        mlps_params.common_params.input_size = [n_patches]
        mlps_params.common_params.num_units = self.mlps_hidden
        return mlps_params

    def _get_cls_params(self):
        cls_params: LinearLayer.ModelParams = LinearLayer.get_params()
        cls_params.common_params.input_size = self.hidden_size
        cls_params.common_params.num_units = 10
        cls_params.common_params.activation = nn.Identity
        return cls_params
    
    def _get_block_params(self):
        mlpc_params = self._get_mlpc_params()
        mlps_params = self._get_mlps_params()
        block_params: MixerBlock.ModelParams = MixerBlock.get_params()
        block_params.channel_mlp_params = mlpc_params
        block_params.spatial_mlp_params = mlps_params
        n_patches = self._get_n_patches()
        block_params.common_params.input_size = [n_patches, self.hidden_size]
        return block_params

    def get_model_params(self):
        patch_params = self._get_patch_params()        
        cls_params = self._get_cls_params()
        block_params = self._get_block_params()

        mixer_params: MLPMixer.ModelParams = MLPMixer.get_params()
        mixer_params.common_params.input_size = self.input_size
        mixer_params.patch_gen_params = patch_params
        mixer_params.mixer_block_params = [deepcopy(block_params) for i in range(self.num_blocks)]
        mixer_params.classifier_params = cls_params
        return mixer_params

class Cifar10AutoAugmentMLPMixer8LTask(Cifar10AutoAugmentMLPMixerTask):
    num_blocks = 8

class Cifar10AutoAugmentMLPMixer1LTask(Cifar10AutoAugmentMLPMixerTask):
    num_blocks = 1

class Cifar10AutoAugmentwRetinaBlurMLPMixerTask(Cifar10AutoAugmentMLPMixerTask):
    def _get_patch_params(self):
        cnn = super()._get_patch_params()
        rblur = RetinaBlurFilter.ModelParams(RetinaBlurFilter, self.input_size, 0.12, 0.06, 0.12, 9)
        norm = NormalizationLayer.ModelParams(NormalizationLayer, [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
        p: SequentialLayers.ModelParams = SequentialLayers.get_params()
        p.common_params.input_size = self.input_size
        p.common_params.activation = nn.Identity
        p.layer_params = [rblur, norm, cnn]
        return p
    
    def get_model_params(self):
        p = super().get_model_params()
        p.normalize_input = False
        return p

    def get_experiment_params(self):
        p = super().get_experiment_params()
        for ap in p.adv_config.testing_attack_params:
            ap.eot_iter = 10
        return p

class Cifar10AutoAugmentwRetinaBlurMLPMixer8LTask(Cifar10AutoAugmentwRetinaBlurMLPMixerTask):
    num_blocks = 8

class Cifar10AutoAugmentwCenteredRetinaBlurMLPMixer8LEvalTask(Cifar10AutoAugmentwRetinaBlurMLPMixer8LTask):
    def _get_patch_params(self):
        p = super()._get_patch_params()
        rblur = p.layer_params[0]
        rblur.loc_mode = 'center'
        return p

class Cifar10AutoAugmentwRetinaBlurMLPMixer1LTask(Cifar10AutoAugmentwRetinaBlurMLPMixerTask):
    num_blocks = 1

class Cifar10AutoAugmentwRetinaSamplerMLPMixerTask(Cifar10AutoAugmentMLPMixerTask):
    def _get_patch_params(self):
        cnn = super()._get_patch_params()
        rblur = RetinaSampleFilter.ModelParams(RetinaSampleFilter, self.input_size, 0.12, 0.09, 0.12, 9)
        norm = NormalizationLayer.ModelParams(NormalizationLayer, [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
        p: SequentialLayers.ModelParams = SequentialLayers.get_params()
        p.common_params.input_size = self.input_size
        p.common_params.activation = nn.Identity
        p.layer_params = [norm, rblur, cnn]
        return p
    
    def get_model_params(self):
        p = super().get_model_params()
        p.normalize_input = False
        return p

    def get_experiment_params(self):
        p = super().get_experiment_params()
        for ap in p.adv_config.testing_attack_params:
            ap.eot_iter = 10
        return p

class Cifar10AutoAugmentwRetinaSamplerMLPMixer8LTask(Cifar10AutoAugmentwRetinaSamplerMLPMixerTask):
    num_blocks = 8

class Cifar10AutoAugmentwCenteredRetinaSamplerMLPMixer8LEvalTask(Cifar10AutoAugmentwRetinaSamplerMLPMixer8LTask):
    def _get_patch_params(self):
        p = super()._get_patch_params()
        rblur = p.layer_params[1]
        rblur.loc_mode = 'center'
        return p
class Cifar10AutoAugmentwRetinaNonUniformPatchEmbeddingMLPMixerTask(Cifar10AutoAugmentMLPMixerTask):
    def _get_patch_params(self):
        rblur = RetinaNonUniformPatchEmbedding.ModelParams(RetinaNonUniformPatchEmbedding, self.input_size, self.hidden_size)
        # rblur.loc_mode = 'center'
        return rblur

    def get_experiment_params(self):
        p = super().get_experiment_params()
        for ap in p.adv_config.testing_attack_params:
            ap.eot_iter = 10
        return p

class Cifar10AutoAugmentwRetinaNonUniformPatchEmbeddingMLPMixer8LTask(Cifar10AutoAugmentwRetinaNonUniformPatchEmbeddingMLPMixerTask):
    num_blocks = 8

class Cifar10AutoAugmentwCenteredRetinaNonUniformPatchEmbeddingMLPMixer8LEvalTask(Cifar10AutoAugmentwRetinaNonUniformPatchEmbeddingMLPMixer8LTask):
    def _get_patch_params(self):
        p = super()._get_patch_params()
        p.loc_mode = 'center'
        return p

class Cifar10AutoAugmentwRetinaNonUniformMaskedPatchEmbeddingMLPMixer8LTask(Cifar10AutoAugmentwRetinaNonUniformPatchEmbeddingMLPMixer8LTask):
    def _get_patch_params(self):
        p = super()._get_patch_params()
        p.mask_small_rf_region = True
        return p
    
    def _get_n_patches(self):
        return 56

class Cifar10AutoAugmentwCenteredRetinaNonUniformMaskedPatchEmbeddingMLPMixer8LEvalTask(Cifar10AutoAugmentwRetinaNonUniformMaskedPatchEmbeddingMLPMixer8LTask):
    def _get_patch_params(self):
        p = super()._get_patch_params()
        p.loc_mode = 'center'
        return p
        
class Cifar10AutoAugmentCAMLPMixerTask(Cifar10AutoAugmentMLPMixerTask):
    mlpc_hidden = 192
    num_steps = 8
    act_opt_lr = 1.

    def _get_patch_params(self):
        p: ScanningConsistentActivationLayer.ModelParams = ScanningConsistentActivationLayer.get_params()
        p.common_params.input_size = [3, 32, 32]
        p.common_params.num_units = self.hidden_size
        p.common_params.activation = nn.Identity
        p.scanning_consistency_optimization_params.kernel_size = self.patch_size
        p.scanning_consistency_optimization_params.stride = self.patch_size
        p.scanning_consistency_optimization_params.padding = 0
        p.scanning_consistency_optimization_params.act_opt_kernel_size = 1
        p.scanning_consistency_optimization_params.act_opt_stride = 1
        p.scanning_consistency_optimization_params.window_input_act_consistency = False
        self._set_co_params(p)
        return p

    def _set_co_params(self, p):
        p.consistency_optimization_params.act_opt_step_size = self.act_opt_lr
        p.consistency_optimization_params.max_train_time_steps = self.num_steps
        p.consistency_optimization_params.max_test_time_steps = self.num_steps
        p.consistency_optimization_params.input_act_consistency = True
        p.consistency_optimization_params.lateral_dependence_type = 'ReLU'
        return p

    def _get_mlpc_params(self):
        p: ConsistentActivationMixerMLP.ModelParams = ConsistentActivationMixerMLP.get_params()
        p.common_params.input_size = self.hidden_size
        p.common_params.activation = nn.ReLU
        p.common_params.num_units = self.mlpc_hidden
        self._set_co_params(p)
        return p
    
    def _get_mlps_params(self):
        n_patches = (self.input_size[1] // self.patch_size)**2
        p: ConsistentActivationMixerMLP.ModelParams = ConsistentActivationMixerMLP.get_params()
        p.common_params.input_size = n_patches
        p.common_params.activation = nn.ReLU
        p.common_params.num_units = self.mlps_hidden
        self._set_co_params(p)
        return p

    def _get_cls_params(self):
        p: FirstNExtractionClassifier.ModelParams = FirstNExtractionClassifier.get_params()
        p.num_classes = 10
        return p

    def _get_block_params(self):
        mlpc_params = self._get_mlpc_params()
        mlps_params = self._get_mlps_params()
        block_params: ConsistentActivationMixerBlock.ModelParams = ConsistentActivationMixerBlock.get_params()
        block_params.channel_mlp_params = mlpc_params
        block_params.spatial_mlp_params = mlps_params
        block_params.consistency_optimization_params
        n_patches = (self.input_size[1] // self.patch_size)**2
        block_params.common_params.input_size = [n_patches, self.hidden_size]
        self._set_co_params(block_params)
        block_params.consistency_optimization_params.lateral_dependence_type = 'Linear'
        return block_params

    def get_experiment_params(self):
        p = ConsistentActivationAdversarialExperimentConfig()
        p.act_opt_config.act_opt_lr_warmup_schedule = ActivityOptimizationSchedule.LINEAR
        p.act_opt_config.init_act_opt_lr = 1e-2
        p.act_opt_config.num_warmup_epochs = 60
        set_common_training_params(p)
        p.training_params.nepochs = 300
        p.optimizer_config = AdamOptimizerConfig()
        p.optimizer_config.weight_decay = 5e-5
        p.scheduler_config = CosineAnnealingWarmRestartsConfig()
        p.training_params.early_stop_patience = 100
        p.training_params.tracked_metric = 'val_accuracy'
        p.training_params.tracking_mode = 'max'
        p.scheduler_config.T_0 = p.training_params.nepochs
        p.scheduler_config.eta_min = 1e-6
        p.batch_size = 128
        test_eps = [0.0, 0.008, 0.016, 0.024, 0.032, 0.048, 0.064]
        set_adv_params(p, test_eps)
        dsp = self.get_dataset_params()
        p.exp_name = f'{self.num_steps}Steps-actOptLR={self.act_opt_lr}-{dsp.max_num_train//1000}K'
        return p
    
class Cifar10AutoAugmentCAMLPMixer1LTask(Cifar10AutoAugmentCAMLPMixerTask):
    num_blocks = 1
    mlpc_hidden = 512
    act_opt_lr = 1.
    num_steps = 8

class Cifar10AutoAugmentCAMLPMixer8LTask(Cifar10AutoAugmentCAMLPMixerTask):
    num_blocks = 8
