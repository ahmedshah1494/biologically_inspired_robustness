from copy import deepcopy
import math
import os
from time import time
from attrs import define, asdict
from mllib.adversarial.attacks import AttackParamFactory, SupportedAttacks, SupportedBackend, AbstractAttackConfig
from mllib.runners.configs import BaseExperimentConfig
import torch
import torchvision

from trainers import RandomizedSmoothingEvaluationTrainer, MultiAttackEvaluationTrainer
from mllib.datasets.dataset_factory import ImageDatasetFactory, SupportedDatasets

from adversarialML.biologically_inspired_models.src.retina_preproc import AbstractRetinaFilter, GaussianNoiseLayer, VOneBlock
from adversarialML.biologically_inspired_models.src.models import GeneralClassifier, LogitAverageEnsembler, XResNet34
from mllib.param import BaseParameters
import numpy as np

def get_pgd_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.PGDLINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.nsteps = 100
    atk_p.step_size = eps/40
    atk_p.random_start = True
    return atk_p

def get_eot10_pgd_atk(eps):
    p = get_pgd_atk(eps)
    p.eot_iter = 10
    return p

def get_eot20_pgd_atk(eps):
    p = get_pgd_atk(eps)
    p.eot_iter = 20
    return p

def get_apgd_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.APGDLINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.nsteps = 100
    return atk_p

def get_eot10_apgd_atk(eps):
    p = get_apgd_atk(eps)
    p.eot_iter = 10
    return p

def get_eot20_apgd_atk(eps):
    p = get_apgd_atk(eps)
    p.eot_iter = 20
    return p

def get_eot50_apgd_atk(eps):
    p = get_apgd_atk(eps)
    p.eot_iter = 50
    return p

def get_apgd_l2_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.APGDL2, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.nsteps = 100
    return atk_p

def get_eot10_apgd_l2_atk(eps):
    p = get_apgd_l2_atk(eps)
    p.eot_iter = 10
    return p

def get_eot20_apgd_l2_atk(eps):
    p = get_apgd_l2_atk(eps)
    p.eot_iter = 20
    return p

def get_eot50_apgd_l2_atk(eps):
    p = get_apgd_l2_atk(eps)
    p.eot_iter = 50
    return p

def get_apgd_l1_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.APGDL1, SupportedBackend.AUTOATTACK)
    atk_p.eps = eps
    atk_p.nsteps = 100
    return atk_p

def get_square_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.SQUARELINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.n_queries = 5000
    atk_p.n_restarts = 2
    atk_p.seed = time()
    return atk_p

def get_randomly_targeted_square_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.RANDOMLY_TARGETED_SQUARELINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    return atk_p

def get_hsj_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.HOPSKIPJUMPLINF, SupportedBackend.FOOLBOX)
    atk_p.eps = eps
    return atk_p

def get_boundary_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.BOUNDARY, SupportedBackend.FOOLBOX)
    atk_p.steps = 1000
    atk_p.run_params.epsilons = [eps]
    return atk_p

def get_transfered_atk(src_model_path, atkfn):
    src_model = torch.load(src_model_path)
    def _atkfn(eps):
        p = atkfn(eps)
        p.model = src_model
        return p
    return _atkfn

def get_cwl2_atk(conf):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.CWL2, SupportedBackend.FOOLBOX)
    atk_p.confidence = conf
    atk_p.steps = 100
    atk_p.run_params.epsilons = [1000.]
    return atk_p

def get_adv_attack_params(atk_types):
    atktype_to_paramfn = {
        SupportedAttacks.PGDLINF: get_pgd_atk,
        SupportedAttacks.APGDLINF: get_apgd_atk,
        SupportedAttacks.SQUARELINF: get_square_atk,
        SupportedAttacks.RANDOMLY_TARGETED_SQUARELINF: get_randomly_targeted_square_atk,
        SupportedAttacks.HOPSKIPJUMPLINF: get_hsj_atk,
    }
    return [atktype_to_paramfn[atkT] for atkT in atk_types]

def set_gaussian_noise_param(p: BaseParameters, param, value):
    if issubclass(p.cls, GaussianNoiseLayer):
        # p.loc_mode = loc_mode
        if hasattr(p, param):
            setattr(p, param, value)
    else:
        d = p.asdict(recurse=False)
        for v in d.values():
            if isinstance(v, BaseParameters):
                set_gaussian_noise_param(v, param, value)
            elif np.iterable(v):
                for x in v:
                    if isinstance(x, BaseParameters):
                        set_gaussian_noise_param(x, param, value)
    return p

def set_retina_param(p: BaseParameters, param, value):
    if issubclass(p.cls, AbstractRetinaFilter):
        # p.loc_mode = loc_mode
        if hasattr(p, param):
            setattr(p, param, value)
    else:
        d = p.asdict(recurse=False)
        for v in d.values():
            if isinstance(v, BaseParameters):
                set_retina_param(v, param, value)
            elif np.iterable(v):
                for x in v:
                    if isinstance(x, BaseParameters):
                        set_retina_param(x, param, value)
    return p

def set_layer_param(p:BaseParameters, layer_cls, param, value):
    if issubclass(p.cls, layer_cls):
        if hasattr(p, param):
            setattr(p, param, value)
    else:
        d = p.asdict(recurse=False)
        for v in d.values():
            if isinstance(v, BaseParameters):
                set_layer_param(v, layer_cls, param, value)
            elif np.iterable(v):
                for x in v:
                    if isinstance(x, BaseParameters):
                        set_layer_param(x, layer_cls, param, value)
    return p

def set_retina_loc_mode(p, loc_mode):
    return set_retina_param(p, 'loc_mode', loc_mode)

def disable_retina_processing(p):
    p = set_retina_param(p, 'no_blur', True)
    p = set_retina_param(p, 'only_color', True)
    return p

def setup_for_five_fixation_evaluation(p: BaseParameters):
    if isinstance(p, GeneralClassifier.ModelParams):
        p.logit_ensembler_params = LogitAverageEnsembler.ModelParams(LogitAverageEnsembler, n=5)
    set_retina_loc_mode(p, 'five_fixations')
    return p

class MultiRandAffineAugments(torch.nn.Module):
    @define(slots=False)
    class ModelParams(BaseParameters):
        n: int = 1
        seeds = [43699768,18187898,89517585,69239841,80428875]
    
    def __init__(self, params: ModelParams):
        super().__init__()
        self.params = params
        shearXYp = math.degrees(0.15)
        shearXYn = math.degrees(-0.15)
        self.affine_augments = torchvision.transforms.RandomAffine(15., (0.22, 0.22), shear=(shearXYn, shearXYp, shearXYn, shearXYp))
    
    def forward(self, img):
        # if 'plt' not in locals():
        #     import matplotlib.pyplot as plt
        #     from adversarialML.biologically_inspired_models.src.retina_preproc import convert_image_tensor_to_ndarray
        rng_state = torch.get_rng_state()
        filtered = []
        for i in range(self.params.n):
            torch.manual_seed(self.params.seeds[i])
            img_ = self.affine_augments(img)
            filtered.append(img_)
        # nrows = nimgs = 3
        # ncols = self.params.n+1
        # for j in range(nimgs):
        #     plt.subplot(nrows, ncols, j*ncols + 1)
        #     plt.imshow(convert_image_tensor_to_ndarray(img[j]))
        #     for i, img_ in enumerate(filtered):
        #         plt.subplot(nrows, ncols,  j*ncols + i + 2)
        #         plt.imshow(convert_image_tensor_to_ndarray(img_[j]))
        # plt.savefig('multi_rand_affine.png')
        filtered = torch.stack(filtered, dim=1)
        filtered = filtered.reshape(-1, *(filtered.shape[2:]))
        torch.set_rng_state(rng_state)
        return filtered

def setup_for_multi_randaugments(p, k):
    mrap = MultiRandAffineAugments.ModelParams(MultiRandAffineAugments, n=k)
    p = set_layer_param(p, XResNet34, 'preprocessing_layer_params', mrap)
    lep = LogitAverageEnsembler.ModelParams(LogitAverageEnsembler, n=k)
    p = set_layer_param(p, XResNet34, 'logit_ensembler_params', lep)
    return p

def get_adversarial_battery_task(task_cls, num_test, batch_size, atk_param_fns, test_eps=[0.0, 0.016, 0.024, 0.032, 0.048, 0.064], 
                                center_fixation=False, five_fixation_ensemble=False, view_scale=None, disable_retina=False, 
                                add_fixed_noise_patch=False, use_common_corruption_testset=False, enable_random_noise=False,
                                apply_rand_affine_augments=False, num_affine_augments=5):
    class AdversarialAttackBatteryEvalTask(task_cls):
        _cls = task_cls
        def get_dataset_params(self):
            p = super().get_dataset_params()
            if p.dataset in [SupportedDatasets.ECOSET, SupportedDatasets.ECOSET100, SupportedDatasets.IMAGENET]:
                p.dataset = {
                    SupportedDatasets.ECOSET: SupportedDatasets.ECOSET_FOLDER,
                    SupportedDatasets.ECOSET100: SupportedDatasets.ECOSET100_FOLDER
                }[p.dataset]
                p.datafolder = f'{os.path.dirname(p.datafolder)}/eval_dataset_dir'
                p.max_num_train = 10000
            if use_common_corruption_testset:
                if p.dataset == SupportedDatasets.ECOSET10:
                    p.dataset = SupportedDatasets.ECOSET10C_FOLDER
                    p.datafolder = os.path.join(os.path.dirname(os.path.dirname(p.datafolder)), 'distorted')
                if p.dataset == SupportedDatasets.ECOSET100_FOLDER:
                    p.dataset = SupportedDatasets.ECOSET100C_FOLDER
                    p.datafolder = os.path.join(p.datafolder, 'distorted')
                if p.dataset == SupportedDatasets.ECOSET_FOLDER:
                    p.dataset = SupportedDatasets.ECOSETC_FOLDER
                    p.datafolder = '/home/mshah1/workhorse3/ecoset/distorted'
                if p.dataset == SupportedDatasets.IMAGENET_FOLDER:
                    p.dataset = SupportedDatasets.ECOSETC_FOLDER
                    p.datafolder = '/home/mshah1/workhorse3/ecoset/distorted'
                if p.dataset == SupportedDatasets.CIFAR10:
                    p.dataset = SupportedDatasets.CIFAR10C
                    p.datafolder = '/home/mshah1/workhorse3/cifar-10-batches-py/distorted'

            p.max_num_test = num_test
            return p
        
        def get_model_params(self):
            p = super().get_model_params()
            if enable_random_noise:
                p = set_gaussian_noise_param(p, 'add_noise_during_inference', True)
                p = set_layer_param(p, VOneBlock, 'add_noise_during_inference', True)
            if apply_rand_affine_augments:
                p = setup_for_multi_randaugments(p, num_affine_augments)
            if disable_retina:
                p = disable_retina_processing(p)
            else:
                p = set_retina_param(p, 'view_scale', view_scale)
                if add_fixed_noise_patch:
                    p = set_gaussian_noise_param(p, 'add_deterministic_noise_during_inference', True)
                    p = set_layer_param(p, VOneBlock, 'add_deterministic_noise_during_inference', True)
                if center_fixation:
                    p = set_retina_loc_mode(p, 'center')
                elif five_fixation_ensemble:
                    p = setup_for_five_fixation_evaluation(p)
            return p

        def get_experiment_params(self) -> BaseExperimentConfig:
            p = super(AdversarialAttackBatteryEvalTask, self).get_experiment_params()
            p.trainer_params.cls = MultiAttackEvaluationTrainer
            p.batch_size = batch_size
            adv_config = p.trainer_params.adversarial_params
            adv_config.training_attack_params = None
            atk_params = []
            for name, atkfn in atk_param_fns.items():
                if apply_rand_affine_augments:
                    name = f'{num_affine_augments}RandAug'
                if use_common_corruption_testset:
                    name = 'CC'+name
                if enable_random_noise:
                    name = 'RandNoise'+name
                if disable_retina:
                    name = f'NoRetina'+name
                else:
                    if add_fixed_noise_patch:
                        name = 'DetNoise'+name
                    if view_scale is not None:
                        name = f'Scale={view_scale}'+name
                    if center_fixation and isinstance(name, str):
                        name = 'Centered'+name
                    if five_fixation_ensemble and isinstance(name, str):
                        name = '5Fixation'+name
                atk_params += [(name, atkfn(eps)) for eps in test_eps]
            adv_config.testing_attack_params = atk_params
            return p
    return AdversarialAttackBatteryEvalTask

def get_randomized_smoothing_task(task_cls, num_test, sigmas, batch_size, rs_batch_size:int = 1000, n0: int = 100, n: int = 100_000, alpha: float = 0.001,
                                    center_fixation=False, five_fixation_ensemble=False, start_idx=0, end_idx=None):
    class RandomizedSmoothingEvalTask(task_cls):
        def get_dataset_params(self):
            p = super().get_dataset_params()
            p.max_num_test = num_test
            return p

        def get_model_params(self):
            p = super().get_model_params()
            p = set_retina_param(p, 'view_scale', None)
            if center_fixation:
                p = set_retina_loc_mode(p, 'center')
            elif five_fixation_ensemble:
                p = setup_for_five_fixation_evaluation(p)
            return p

        def get_experiment_params(self) -> BaseExperimentConfig:
            keys_to_keep = [a.name for a in RandomizedSmoothingEvaluationTrainer.TrainerParams.__attrs_attrs__]
            p = super(RandomizedSmoothingEvalTask, self).get_experiment_params()
            p_dict = asdict(p.trainer_params, recurse=False, filter=lambda a,v: (a.name in keys_to_keep))
            p.trainer_params = RandomizedSmoothingEvaluationTrainer.TrainerParams(RandomizedSmoothingEvaluationTrainer, p.trainer_params.training_params)
            # p.logdir = '/share/workhorse3/mshah1/biologically_inspired_models/logs/'
            p.trainer_params.randomized_smoothing_params.num_classes = ImageDatasetFactory.dataset_config[self.get_dataset_params().dataset].nclasses
            p.trainer_params.randomized_smoothing_params.sigmas = sigmas
            p.trainer_params.randomized_smoothing_params.batch = rs_batch_size
            p.trainer_params.randomized_smoothing_params.N0 = n0
            p.trainer_params.randomized_smoothing_params.N = n
            p.trainer_params.randomized_smoothing_params.alpha = alpha
            p.trainer_params.randomized_smoothing_params.start_idx = start_idx
            p.trainer_params.randomized_smoothing_params.end_idx = end_idx if end_idx is not None else num_test
            p.batch_size = num_test
            if center_fixation:
                p.trainer_params.exp_name = 'Centered-'+p.trainer_params.exp_name
            if five_fixation_ensemble:
                p.trainer_params.exp_name = '5Fixation-'+p.trainer_params.exp_name
            return p
    return RandomizedSmoothingEvalTask
# def add_adversarial_attack_battery_config(training_task: AbstractTask):
#     training_task.get_experiment_params = MethodType(adversarial_attack_battery_config, training_task)
