from typing import List
from attrs import define
from mllib.runners.base_runners import BaseRunner
from mllib.runners.configs import BaseExperimentConfig
from mllib.tasks.base_tasks import AbstractTask

from trainers import ActivityOptimizationParams, AdversarialTrainer, AdversarialParams, ConsistentActivationModelAdversarialTrainer

@define(slots=False)
class AdversarialExperimentConfig(BaseExperimentConfig):
    adv_config: AdversarialParams = AdversarialParams()

@define(slots=False)
class ConsistentActivationAdversarialExperimentConfig(AdversarialExperimentConfig):
    act_opt_config: ActivityOptimizationParams = ActivityOptimizationParams()

class AdversarialExperimentRunner(BaseRunner):
    def create_trainer(self):
        trainer_params = self.create_trainer_params()
        adv_trainer_params = AdversarialTrainer.get_params()
        adv_trainer_params.trainer_params = trainer_params
        adv_trainer_params.adversarial_params = self.task.get_experiment_params().adv_config
        self.trainer = adv_trainer_params.cls(adv_trainer_params)

class ConsistentActivationAdversarialExperimentRunner(AdversarialExperimentRunner):
    def create_trainer(self):
        trainer_params = self.create_trainer_params()
        adv_trainer_params = ConsistentActivationModelAdversarialTrainer.get_params()
        adv_trainer_params.trainer_params = trainer_params
        adv_trainer_params.adversarial_params = self.task.get_experiment_params().adv_config
        adv_trainer_params.act_opt_params = self.task.get_experiment_params().act_opt_config
        self.trainer = adv_trainer_params.cls(adv_trainer_params)