from attrs import define
from mllib.adversarial.attacks import AttackParamFactory, SupportedAttacks, SupportedBackend
from mllib.runners.configs import BaseExperimentConfig
import torch

def get_pgd_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.PGDLINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.nsteps = 100
    atk_p.step_size = eps/40
    atk_p.random_start = True
    return atk_p

def get_apgd_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.APGDLINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    atk_p.nsteps = 100
    return atk_p

def get_square_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.SQUARELINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    return atk_p

def get_randomly_targeted_square_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.RANDOMLY_TARGETED_SQUARELINF, SupportedBackend.TORCHATTACKS)
    atk_p.eps = eps
    return atk_p

def get_hsj_atk(eps):
    atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.HOPSKIPJUMPLINF, SupportedBackend.FOOLBOX)
    atk_p.eps = eps
    return atk_p

def get_transfered_atk(src_model_path, atkfn):
    src_model = torch.load(src_model_path)
    def _atkfn(eps):
        p = atkfn(eps)
        p.model = src_model
        return p

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
            p.batch_size = batch_size
            adv_config = p.adv_config
            adv_config.training_attack_params = None
            atk_params = []
            for atkfn in atk_param_fns:
                atk_params += [atkfn(eps) for eps in test_eps]
            adv_config.testing_attack_params = atk_params
            return p
    return AdversarialAttackBatteryEvalTask

# def add_adversarial_attack_battery_config(training_task: AbstractTask):
#     training_task.get_experiment_params = MethodType(adversarial_attack_battery_config, training_task)
