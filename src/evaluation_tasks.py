from mllib.tasks.base_tasks import AbstractTask
from mllib.adversarial.attacks import AttackParamFactory, SupportedAttacks, SupportedBackend
from mllib.runners.configs import BaseExperimentConfig

from runners import AdversarialExperimentConfig
from types import MethodType

from tasks import get_cifar10_params

def get_adversarial_battery_task(task_cls):
    class AdversarialAttackBatteryEvalTask(task_cls):
        def get_dataset_params(self):
            return get_cifar10_params(num_test=128)

        def get_experiment_params(self) -> BaseExperimentConfig:
            p = super(AdversarialAttackBatteryEvalTask, self).get_experiment_params()
            p.batch_size = 128
            adv_config = p.adv_config
            adv_config.training_attack_params = None
            test_eps = [0.0, 0.016, 0.024, 0.032, 0.048, 0.064]
            
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
            
            def get_hsj_atk(eps):
                atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.HOPSKIPJUMPLINF, SupportedBackend.FOOLBOX)
                atk_p.eps = eps
                return atk_p

            atk_params = []
            for atkfn in [get_apgd_atk]:
                atk_params += [atkfn(eps) for eps in test_eps]
            adv_config.testing_attack_params = atk_params
            return p
    return AdversarialAttackBatteryEvalTask

# def add_adversarial_attack_battery_config(training_task: AbstractTask):
#     training_task.get_experiment_params = MethodType(adversarial_attack_battery_config, training_task)
