from time import time
from attrs import define, asdict
from mllib.adversarial.attacks import AttackParamFactory, SupportedAttacks, SupportedBackend
from mllib.runners.configs import BaseExperimentConfig
import torch

from adversarialML.biologically_inspired_models.src.trainers import RandomizedSmoothingEvaluationTrainer, MultiAttackEvaluationTrainer

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


def get_adversarial_battery_task(task_cls, num_test, batch_size, atk_param_fns, test_eps=[0.0, 0.016, 0.024, 0.032, 0.048, 0.064]):
    class AdversarialAttackBatteryEvalTask(task_cls):
        def get_dataset_params(self):
            p = super().get_dataset_params()
            p.max_num_test = num_test
            return p

        def get_experiment_params(self) -> BaseExperimentConfig:
            p = super(AdversarialAttackBatteryEvalTask, self).get_experiment_params()
            p.trainer_params.cls = MultiAttackEvaluationTrainer
            p.batch_size = batch_size
            adv_config = p.trainer_params.adversarial_params
            adv_config.training_attack_params = None
            atk_params = []
            for name, atkfn in atk_param_fns.items():
                atk_params += [(name, atkfn(eps)) for eps in test_eps]
            adv_config.testing_attack_params = atk_params
            return p
    return AdversarialAttackBatteryEvalTask

def get_randomized_smoothing_task(task_cls, num_test, sigmas, batch_size, rs_batch_size:int = 1000, n0: int = 100, n: int = 100_000, alpha: float = 0.001):
    class RandomizedSmoothingEvalTask(task_cls):
        def get_dataset_params(self):
            p = super().get_dataset_params()
            p.max_num_test = num_test
            return p

        def get_experiment_params(self) -> BaseExperimentConfig:
            keys_to_keep = [a.name for a in RandomizedSmoothingEvaluationTrainer.TrainerParams.__attrs_attrs__]
            p = super(RandomizedSmoothingEvalTask, self).get_experiment_params()
            p_dict = asdict(p.trainer_params, recurse=False, filter=lambda a,v: (a.name in keys_to_keep))
            p.trainer_params = RandomizedSmoothingEvaluationTrainer.TrainerParams(RandomizedSmoothingEvaluationTrainer, p.trainer_params.training_params)
            p.logdir = '/share/workhorse3/mshah1/biologically_inspired_models/logs/'
            p.trainer_params.randomized_smoothing_params.num_classes = 10
            p.trainer_params.randomized_smoothing_params.sigmas = sigmas
            p.trainer_params.randomized_smoothing_params.batch = rs_batch_size
            p.trainer_params.randomized_smoothing_params.N0 = n0
            p.trainer_params.randomized_smoothing_params.N = n
            p.trainer_params.randomized_smoothing_params.alpha = alpha
            p.batch_size = batch_size
            return p
    return RandomizedSmoothingEvalTask
# def add_adversarial_attack_battery_config(training_task: AbstractTask):
#     training_task.get_experiment_params = MethodType(adversarial_attack_battery_config, training_task)
