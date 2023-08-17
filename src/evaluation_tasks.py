from copy import deepcopy
import math
import os
from time import time
from attr import field
from attrs import define, asdict
from mllib.adversarial.attacks import AttackParamFactory, SupportedAttacks, SupportedBackend, AbstractAttackConfig
from mllib.runners.configs import BaseExperimentConfig
import torch
import torchvision
from adversarialML.biologically_inspired_models.src.fixation_prediction.precomputed_fixation_attacks import PrecomputedFixationAPGDAttack

from trainers import RandomizedSmoothingEvaluationTrainer, MultiAttackEvaluationTrainer, AnnotatedMultiAttackEvaluationTrainer
from fixation_prediction.trainers import PrecomputedFixationMapMultiAttackEvaluationTrainer, RetinaFilterWithFixationPredictionMultiAttackEvaluationTrainer
from mllib.datasets.dataset_factory import ImageDatasetFactory, SupportedDatasets

from adversarialML.biologically_inspired_models.src.fixation_prediction.models import RetinaFilterWithFixationPrediction, MultiFixationTiedBackboneClassifier
from adversarialML.biologically_inspired_models.src.retina_preproc import AbstractRetinaFilter, GaussianNoiseLayer, VOneBlock
from adversarialML.biologically_inspired_models.src.models import GeneralClassifier, LogitAverageEnsembler, XResNet34, IdentityLayer, CommonModelParams, MultiheadSelfAttentionEnsembler
from adversarialML.biologically_inspired_models.src.fixation_prediction.fixation_aware_attack import FixationAwareAPGDAttack
from mllib.param import BaseParameters
import numpy as np

def get_pgd_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.PGDLINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.nsteps = 100
    atk_p.step_size = eps/40
    atk_p.random_start = True
    return atk_p

def get_pgd_75s_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.PGDLINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.nsteps = 75
    atk_p.step_size = eps/30
    atk_p.random_start = True
    return atk_p

def get_pgd_50s_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.PGDLINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.nsteps = 50
    atk_p.step_size = eps/20
    atk_p.random_start = True
    return atk_p

def get_pgd_25s_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.PGDLINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.nsteps = 25
    atk_p.step_size = eps/10
    atk_p.random_start = True
    atk_p.verbose = True
    return atk_p

def get_pgd_10s_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.PGDLINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.nsteps = 10
    atk_p.step_size = eps/4
    atk_p.random_start = True
    atk_p.verbose = True
    return atk_p

def get_pgd_5s_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.PGDLINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.nsteps = 5
    atk_p.step_size = eps
    atk_p.random_start = True
    atk_p.verbose = True
    return atk_p

def get_pgd_1s_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.PGDLINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.nsteps = 1
    atk_p.step_size = eps
    atk_p.random_start = True
    atk_p.verbose = True
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

def get_apgd_75s_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.APGDLINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.nsteps = 75
    return atk_p

def get_apgd_50s_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.APGDLINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.nsteps = 50
    return atk_p

def get_apgd_25s_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.APGDLINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.nsteps = 25
    return atk_p

def get_apgd_10s_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.APGDLINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.nsteps = 10
    return atk_p

def get_apgd_5s_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.APGDLINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.nsteps = 5
    return atk_p

def get_apgd_1s_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.APGDLINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.nsteps = 1
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

def get_apgd_l2_25s_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.APGDL2, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.nsteps = 25
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

def get_autoattack_linf_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.AUTOATTACK, SupportedBackend.AUTOATTACK)
    atk_p.norm = 'Linf'
    atk_p.eps = eps
    return atk_p

def get_autoattack_l2_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.AUTOATTACK, SupportedBackend.AUTOATTACK)
    atk_p.norm = 'L2'
    atk_p.eps = eps
    return atk_p

def get_torchattack_autoattack_linf_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.AUTOATTACK, SupportedBackend.TORCHATTACKS)
    atk_p.norm = 'Linf'
    atk_p.eps = eps
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


@define(slots=False)
class FixationAwareAPGDAttackParams(AbstractAttackConfig):
    _cls = FixationAwareAPGDAttack
    num_fixation_samples: int = 50
    fixation_selection_attack_num_steps: int = 20
    norm: str = 'Linf'
    eps: float = 8/255
    nsteps: int = 50
    n_restarts: int = 2
    seed: int = field(factory=lambda : int(time()))
    loss: str = 'ce'
    eot_iter: int = 1
    rho: float = .75
    verbose:bool=True

    def asdict(self):
        d = super().asdict()
        d['steps'] = d.pop('nsteps')
        return d
        
def get_fixation_aware_atk(eps):    
    atk_p = FixationAwareAPGDAttackParams()
    atk_p.eps = eps
    return atk_p

def get_precomputed_fixation_apgd_atk(eps):
    atk_p = get_apgd_atk(eps)
    atk_p._cls = PrecomputedFixationAPGDAttack
    return atk_p

def get_precomputed_fixation_apgd_1s_atk(eps):
    atk_p = get_apgd_1s_atk(eps)
    atk_p._cls = PrecomputedFixationAPGDAttack
    return atk_p

def get_precomputed_fixation_apgd_5s_atk(eps):
    atk_p = get_apgd_5s_atk(eps)
    atk_p._cls = PrecomputedFixationAPGDAttack
    return atk_p

def get_precomputed_fixation_apgd_10s_atk(eps):
    atk_p = get_apgd_10s_atk(eps)
    atk_p._cls = PrecomputedFixationAPGDAttack
    return atk_p

def get_precomputed_fixation_apgd_25s_atk(eps):
    atk_p = get_apgd_25s_atk(eps)
    atk_p._cls = PrecomputedFixationAPGDAttack
    return atk_p

def get_precomputed_fixation_eot10_apgd_25s_atk(eps):
    atk_p = get_apgd_25s_atk(eps)
    atk_p._cls = PrecomputedFixationAPGDAttack
    atk_p.eot_iter = 10
    return atk_p

def get_precomputed_fixation_apgd_50s_atk(eps):
    atk_p = get_apgd_50s_atk(eps)
    atk_p._cls = PrecomputedFixationAPGDAttack
    return atk_p

def get_precomputed_fixation_apgd_75s_atk(eps):
    atk_p = get_apgd_75s_atk(eps)
    atk_p._cls = PrecomputedFixationAPGDAttack
    return atk_p

def get_precomputed_fixation_apgd_l2_atk(eps):
    atk_p = get_apgd_l2_atk(eps)
    atk_p._cls = PrecomputedFixationAPGDAttack
    return atk_p

def get_precomputed_fixation_apgd_l2_25s_atk(eps):
    atk_p = get_apgd_l2_25s_atk(eps)
    atk_p._cls = PrecomputedFixationAPGDAttack
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

def set_general_classifier_param(p: BaseParameters, param, value):
    if issubclass(p.cls, GeneralClassifier):
        # p.loc_mode = loc_mode
        if hasattr(p, param):
            setattr(p, param, value)
    else:
        d = p.asdict(recurse=False)
        for v in d.values():
            if isinstance(v, BaseParameters):
                set_general_classifier_param(v, param, value)
            elif np.iterable(v):
                for x in v:
                    if isinstance(x, BaseParameters):
                        set_general_classifier_param(x, param, value)
    return p

def set_param(p:BaseParameters, param, value):
    if hasattr(p, param):
        setattr(p, param, value)
    else:
        d = p.asdict(recurse=False)
        for v in d.values():
            if isinstance(v, BaseParameters):
                set_param(v, param, value)
            elif np.iterable(v):
                for x in v:
                    if isinstance(x, BaseParameters):
                        set_param(x, param, value)
    return p

def get_param(p:BaseParameters, param_name=None, param_type=None, default=None):
    if (param_name is not None) and hasattr(p, param_name):
        return getattr(p, param_name)
    elif (param_type is not None) and isinstance(p, param_type):
        return p
    else:
        p_ = None
        d = p.asdict(recurse=False)
        for v in d.values():
            if isinstance(v, BaseParameters):
                p_ = get_param(v, param_name, param_type, default)
                if p_ is not None:
                    return p_
            elif np.iterable(v):
                for x in v:
                    if isinstance(x, BaseParameters):
                        p_ = get_param(x, param_name, param_type, default)
                        if p_ is not None:
                            return p_
    return p_

def set_param2(p:BaseParameters, value, param_name=None, param_type=None, default=None, set_all=False):
    # print(p)
    if (param_name is not None) and hasattr(p, param_name):
        setattr(p, param_name, value)
        if not set_all:
            return True

    param_set = False
    if hasattr(p, 'asdict'):
        d = p.asdict(recurse=False)
        for k,v in d.items():            
            if np.iterable(v):
                for i,x in enumerate(v):
                    if isinstance(x, BaseParameters):
                        if (param_type is not None) and isinstance(x, param_type):
                            v[i] = value
                            param_set = p
                        else:
                            param_set = set_param2(x, value, param_name, param_type, default, set_all)
                        if param_set and (not set_all):
                            return param_set
            else:
                if (param_type is not None) and isinstance(v, param_type):
                    setattr(p, k, value)
                    param_set = True
                else:
                    param_set = set_param2(v, value, param_name, param_type, default, set_all)
                if param_set and (not set_all):
                    return param_set
    return param_set

def setup_for_five_fixation_evaluation(p: BaseParameters):
    if get_param(p, 'feature_ensembler_params') is None:
        ensembler_params = LogitAverageEnsembler.ModelParams(LogitAverageEnsembler, n=5)
        set_param(p, 'logit_ensembler_params', ensembler_params)
    fpn_params = get_param(p, param_type=RetinaFilterWithFixationPrediction.ModelParams)
    print(fpn_params)
    if fpn_params is not None:
        set_param(fpn_params, 'num_eval_fixation_points', 5)
        set_retina_loc_mode(p, 'const')
    else:
        set_retina_loc_mode(p, 'five_fixations')
    return p

def set_num_fixations(p: BaseParameters, n):
    if (n > 1) and not isinstance(p, MultiFixationTiedBackboneClassifier.ModelParams):
        if get_param(p, 'feature_ensembler_params') is None:
            ensembler_params = LogitAverageEnsembler.ModelParams(LogitAverageEnsembler, n=n)
            set_param(p, 'logit_ensembler_params', ensembler_params)
    fpn_params = get_param(p, param_type=RetinaFilterWithFixationPrediction.ModelParams)
    if fpn_params is not None:
        set_param(fpn_params, 'num_eval_fixation_points', n)
        set_retina_loc_mode(p, 'const')
    print(fpn_params)
    return p

def setup_for_hscan_fixation_evaluation(p: BaseParameters):
    ensembler_params = LogitAverageEnsembler.ModelParams(LogitAverageEnsembler, n=5)
    set_param(p, 'logit_ensembler_params', ensembler_params)
    set_retina_loc_mode(p, 'hscan_fixations')
    return p

def setup_for_fixation_selection(p: BaseParameters):
    retinap = get_param(p, param_type=AbstractRetinaFilter.ModelParams)
    noisep = get_param(p, param_type=GaussianNoiseLayer.ModelParams, default=IdentityLayer.get_params())
    retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,
                                                                    CommonModelParams([4, *(retinap.input_shape[1:])]), 
                                                                    noisep, retinap, IdentityLayer.get_params(),
                                                                    target_downsample_factor=32, apply_retina_before_fixation=False,
                                                                    salience_map_provided_as_input_channel=True,
                                                                    loc_sampling_temp=1.)
    set_param2(p, retinafixp, param_type=AbstractRetinaFilter.ModelParams)
    set_param2(p, IdentityLayer.get_params(), param_type=GaussianNoiseLayer.ModelParams)
    set_param2(p, [4, *(retinap.input_shape[1:])], param_name='input_size', set_all=True)        
    return p

def add_fixation_predictor_to_model(p, fixation_prediction_model_name, apply_retina_before_fixation=False):
    model_name_split = fixation_prediction_model_name.split(':')
    if model_name_split[0] == 'deepgazeII':
        if len(model_name_split) == 0:
            from adversarialML.biologically_inspired_models.src.fixation_prediction.models import DeepGazeII
            fpm = DeepGazeII.get_params()
        else:
            raise NotImplementedError('custom configurations for deepgaze-II are not supported at this time.')
    elif model_name_split[0] == 'deepgazeIII':
        if len(model_name_split) == 0:
            from adversarialML.biologically_inspired_models.src.fixation_prediction.models import DeepGazeIII
            fpm = DeepGazeIII.get_params()
        else:
            from adversarialML.biologically_inspired_models.src.fixation_prediction.models import CustomBackboneDeepGazeIII
            fpm = CustomBackboneDeepGazeIII.get_pretrained_model_params(model_name_split[1])
    else:
        raise NotImplementedError(f'Expected arch to be deepgazeII or deepgazeIII, but got {model_name_split[0]}')
    retinap = get_param(p, param_type=AbstractRetinaFilter.ModelParams)
    noisep = get_param(p, param_type=GaussianNoiseLayer.ModelParams, default=IdentityLayer.get_params())
    retinafixp = RetinaFilterWithFixationPrediction.ModelParams(RetinaFilterWithFixationPrediction,
                                                                    CommonModelParams([4, *(retinap.input_shape[1:])]), 
                                                                    noisep, retinap, fixation_params=fpm,
                                                                    target_downsample_factor=1,
                                                                    loc_sampling_temp=1.,
                                                                    apply_retina_before_fixation=apply_retina_before_fixation)
    set_param2(p, retinafixp, param_type=AbstractRetinaFilter.ModelParams)
    set_param2(p, IdentityLayer.get_params(), param_type=GaussianNoiseLayer.ModelParams)
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
                                center_fixation=False, five_fixation_ensemble=False, hscan_fixation_ensemble=False, view_scale=None, disable_retina=False, 
                                add_fixed_noise_patch=False, use_common_corruption_testset=False, disable_reconstruction=False, use_residual_img=False,
                                fixate_in_bbox=False, enable_random_noise=False, apply_rand_affine_augments=False, num_affine_augments=5, fixate_on_max_loc=False,
                                clickme_data=False, use_precomputed_fixations=False, num_fixations=1, precompute_fixation_map=False, add_fixation_predictor=False,
                                retina_after_fixation=False, fixation_prediction_model='deepgazeII', straight_through_retina=False):
    class AdversarialAttackBatteryEvalTask(task_cls):
        _cls = task_cls
        def __init__(self) -> None:
            self.has_fixation_prediction_network = get_param(self.get_model_params(), param_type=RetinaFilterWithFixationPrediction.ModelParams)
            # self.use_precomputed_fixation_maps = self.get_dataset_params().dataset in [SupportedDatasets.CLICKME,
            #                    SupportedDatasets.ECOSET10wFIXATIONMAPS_FOLDER,
            #                    SupportedDatasets.ECOSET100wFIXATIONMAPS_FOLDER] and get_param(self.get_dataset_params(), param_type=AbstractRetinaFilter.ModelParams)
        
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
                if p.dataset in [SupportedDatasets.ECOSET10, SupportedDatasets.ECOSET10_FOLDER]:
                    p.dataset = SupportedDatasets.ECOSET10C_FOLDER
                    # p.datafolder = os.path.join(os.path.dirname(os.path.dirname(p.datafolder)), 'distorted')
                    p.datafolder = '/home/mshah1/workhorse3/ecoset-10/distorted'
                if p.dataset == SupportedDatasets.ECOSET100_FOLDER:
                    p.dataset = SupportedDatasets.ECOSET100C_FOLDER
                    # p.datafolder = os.path.join(p.datafolder, 'distorted')
                    p.datafolder = '/home/mshah1/workhorse3/ecoset-100/distorted'
                if p.dataset == SupportedDatasets.ECOSET_FOLDER:
                    p.dataset = SupportedDatasets.ECOSETC_FOLDER
                    p.datafolder = '/home/mshah1/workhorse3/ecoset/distorted/'
                    # p.dataset = SupportedDatasets.ECOSETC
                    # p.datafolder = '/home/mshah1/workhorse3/ecoset/distorted/shards'
                if p.dataset == SupportedDatasets.IMAGENET_FOLDER:
                    p.dataset = SupportedDatasets.IMAGENETC_FOLDER
                    p.datafolder = '/home/mshah1/workhorse3/imagenet/distorted'
                if p.dataset == SupportedDatasets.CIFAR10:
                    p.dataset = SupportedDatasets.CIFAR10C
                    p.datafolder = '/home/mshah1/workhorse3/cifar-10-batches-py/distorted'
            if fixate_in_bbox:
                if p.dataset == SupportedDatasets.ECOSET10:
                    p.dataset = SupportedDatasets.ECOSET10wBB_FOLDER
                    p.datafolder = os.path.dirname(os.path.dirname(p.datafolder))
            if clickme_data:
                assert (p.dataset == SupportedDatasets.IMAGENET) or (p.dataset == SupportedDatasets.IMAGENET_FOLDER)
                p.dataset = SupportedDatasets.CLICKME
                p.datafolder = '/share/workhorse3/mshah1/clickme/shards'

            p.max_num_test = num_test
            return p
        
        def get_model_params(self):
            p = super().get_model_params()
            if use_precomputed_fixations and (not fixate_on_max_loc):
                p = setup_for_fixation_selection(p)
            elif add_fixation_predictor:
                p = add_fixation_predictor_to_model(p, fixation_prediction_model, apply_retina_before_fixation=(not retina_after_fixation))
            if enable_random_noise:
                p = set_gaussian_noise_param(p, 'add_noise_during_inference', True)
                p = set_layer_param(p, VOneBlock, 'add_noise_during_inference', True)
            else:
                p = set_gaussian_noise_param(p, 'add_noise_during_inference', False)
                p = set_layer_param(p, VOneBlock, 'add_noise_during_inference', False)
            if add_fixed_noise_patch:
                p = set_gaussian_noise_param(p, 'add_deterministic_noise_during_inference', True)
                p = set_layer_param(p, VOneBlock, 'add_deterministic_noise_during_inference', True)
            else:
                p = set_gaussian_noise_param(p, 'add_deterministic_noise_during_inference', False)
                p = set_layer_param(p, VOneBlock, 'add_deterministic_noise_during_inference', False)
                print(p)
            if apply_rand_affine_augments:
                p = setup_for_multi_randaugments(p, num_affine_augments)
            if disable_retina:
                p = disable_retina_processing(p)
            else:
                p = set_retina_param(p, 'view_scale', view_scale)
                p = set_retina_param(p, 'straight_through', straight_through_retina)
                if center_fixation:
                    p = set_retina_loc_mode(p, 'center')
                elif five_fixation_ensemble:
                    p = setup_for_five_fixation_evaluation(p)
                elif hscan_fixation_ensemble:
                    p = setup_for_hscan_fixation_evaluation(p)
                else:
                    p = set_num_fixations(p, num_fixations)
            if disable_reconstruction:
                p = set_param(p, 'no_reconstruction', True)
            if use_residual_img:
                p = set_param(p, 'use_residual_during_inference', True)
            print(p)
            return p

        def get_experiment_params(self) -> BaseExperimentConfig:
            p = super(AdversarialAttackBatteryEvalTask, self).get_experiment_params()
            dp = super().get_dataset_params()
            if fixate_in_bbox:
                p.trainer_params.cls = AnnotatedMultiAttackEvaluationTrainer
            elif self.get_dataset_params().dataset in [SupportedDatasets.CLICKME,
                               SupportedDatasets.ECOSET10wFIXATIONMAPS_FOLDER,
                               SupportedDatasets.ECOSET100wFIXATIONMAPS_FOLDER]:
                p.trainer_params.cls = PrecomputedFixationMapMultiAttackEvaluationTrainer
                p.trainer_params.set_fixation_to_max = fixate_on_max_loc
            elif self.has_fixation_prediction_network and precompute_fixation_map:
                p.trainer_params.cls = RetinaFilterWithFixationPredictionMultiAttackEvaluationTrainer
            else:
                p.trainer_params.cls = MultiAttackEvaluationTrainer
            p.batch_size = batch_size
            adv_config = p.trainer_params.adversarial_params
            adv_config.training_attack_params = None
            atk_params = []
            for name, atkfn in atk_param_fns.items():
                if apply_rand_affine_augments:
                    name = f'{num_affine_augments}RandAug'+ name
                if clickme_data:
                    name = 'ClickMeData'+name
                if use_common_corruption_testset:
                    name = 'CC'+name
                if use_precomputed_fixations or precompute_fixation_map:
                    name = 'PrecomputedFmap'+name
                if disable_reconstruction:
                    name = f'NoRecon'+name
                if use_residual_img:
                    name = 'ResidualImg'+name
                if enable_random_noise:
                    name = 'RandNoise'+name
                if add_fixation_predictor:
                    name = f'{fixation_prediction_model}Fixations'+name
                if disable_retina:
                    name = f'NoRetina'+name
                else:
                    if add_fixed_noise_patch:
                        name = 'DetNoise'+name
                    if view_scale is not None:
                        name = f'Scale={view_scale}'+name
                    if straight_through_retina:
                        name = 'StraightThrough'+name
                    if center_fixation and isinstance(name, str):
                        name = 'Centered'+name
                    elif five_fixation_ensemble and isinstance(name, str):
                        name = '5Fixation'+name
                    elif hscan_fixation_ensemble and isinstance(name, str):
                        name = 'HscanFixation'+name
                    elif fixate_in_bbox:
                        name = 'BBFixation'+name
                    elif fixate_on_max_loc:
                        name = 'Top1Fixation'+name
                    elif use_precomputed_fixations or precompute_fixation_map or self.has_fixation_prediction_network:
                        name = f'Top{num_fixations}Fixation{"s" if num_fixations > 1 else ""}'+name
                atk_params += [(name, atkfn(eps)) for eps in test_eps]
            adv_config.testing_attack_params = atk_params
            return p
    return AdversarialAttackBatteryEvalTask

def get_randomized_smoothing_task(task_cls, num_test, sigmas, batch_size, rs_batch_size:int = 1000, n0: int = 100, n: int = 100_000, alpha: float = 0.001,
                                    center_fixation=False, five_fixation_ensemble=False, start_idx=0, end_idx=None, add_fixed_noise_patch=False,):
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
            
            if add_fixed_noise_patch:
                p = set_gaussian_noise_param(p, 'add_deterministic_noise_during_inference', True)
                p = set_layer_param(p, VOneBlock, 'add_deterministic_noise_during_inference', True)
            else:
                p = set_gaussian_noise_param(p, 'add_deterministic_noise_during_inference', False)
                p = set_layer_param(p, VOneBlock, 'add_deterministic_noise_during_inference', False)
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
            if add_fixed_noise_patch:
                p.trainer_params.exp_name = 'DetNoise'+p.trainer_params.exp_name
            if center_fixation:
                p.trainer_params.exp_name = 'Centered-'+p.trainer_params.exp_name
            if five_fixation_ensemble:
                p.trainer_params.exp_name = '5Fixation-'+p.trainer_params.exp_name
            return p
    return RandomizedSmoothingEvalTask
# def add_adversarial_attack_battery_config(training_task: AbstractTask):
#     training_task.get_experiment_params = MethodType(adversarial_attack_battery_config, training_task)
