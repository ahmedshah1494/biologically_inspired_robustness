from copy import deepcopy
import os
from time import time
from attrs import define, asdict
from mllib.adversarial.attacks import AttackParamFactory, SupportedAttacks, SupportedBackend, AbstractAttackConfig
from mllib.runners.configs import BaseExperimentConfig
import torch
import torchvision

from trainers import RandomizedSmoothingEvaluationTrainer, MultiAttackEvaluationTrainer
from mllib.datasets.dataset_factory import ImageDatasetFactory, SupportedDatasets

from adversarialML.biologically_inspired_models.src.retina_preproc import AbstractRetinaFilter, GaussianNoiseLayer
from adversarialML.biologically_inspired_models.src.models import GeneralClassifier, LogitAverageEnsembler
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

def get_adversarial_battery_task(task_cls, num_test, batch_size, atk_param_fns, test_eps=[0.0, 0.016, 0.024, 0.032, 0.048, 0.064], 
                                center_fixation=False, five_fixation_ensemble=False, view_scale=None, disable_retina=False, 
                                add_fixed_noise_patch=False, use_common_corruption_testset=False):
    class AdversarialAttackBatteryEvalTask(task_cls):
        _cls = task_cls
        def get_dataset_params(self):
            p = super().get_dataset_params()
            if p.dataset == SupportedDatasets.ECOSET:
                p.dataset = SupportedDatasets.ECOSET_FOLDER
                p.datafolder = os.path.dirname(os.path.dirname(p.datafolder))
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
                if p.dataset == SupportedDatasets.CIFAR10:
                    p.dataset = SupportedDatasets.CIFAR10C
                    p.datafolder = '/home/mshah1/workhorse3/cifar-10-batches-py/distorted'

            p.max_num_test = num_test
            return p
        
        def get_model_params(self):
            p = super().get_model_params()
            if disable_retina:
                p = disable_retina_processing(p)
            else:
                p = set_retina_param(p, 'view_scale', view_scale)
                if add_fixed_noise_patch:
                    p = set_gaussian_noise_param(p, 'add_deterministic_noise_during_inference', True)
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
                if use_common_corruption_testset:
                    name = 'CC'+name
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
                                    center_fixation=False, five_fixation_ensemble=False):
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
            p.batch_size = batch_size
            if center_fixation:
                p.trainer_params.exp_name = 'Centered-'+p.trainer_params.exp_name
            if five_fixation_ensemble:
                p.trainer_params.exp_name = '5Fixation-'+p.trainer_params.exp_name
            return p
    return RandomizedSmoothingEvalTask
# def add_adversarial_attack_battery_config(training_task: AbstractTask):
#     training_task.get_experiment_params = MethodType(adversarial_attack_battery_config, training_task)
