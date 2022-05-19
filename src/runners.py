from typing import List
from attrs import define
from mllib.runners.base_runners import BaseRunner
from mllib.runners.configs import BaseExperimentConfig
from mllib.tasks.base_tasks import AbstractTask

from trainers import AdversarialTrainer, AdversarialParams

@define(slots=False)
class AdversarialExperimentConfig(BaseExperimentConfig):
    adv_config: AdversarialParams = AdversarialParams()

class AdversarialExperimentRunner(BaseRunner):
    def __init__(self, task: AbstractTask) -> None:
        super().__init__(task)

    def create_trainer(self):
        trainer_params = self.create_trainer_params()
        adv_trainer_params = AdversarialTrainer.get_params()
        adv_trainer_params.trainer_params = trainer_params
        adv_trainer_params.adversarial_params = self.task.get_experiment_params().adv_config
        self.trainer = adv_trainer_params.cls(adv_trainer_params)