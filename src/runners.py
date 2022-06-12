import os
import shutil
from typing import List
from attrs import define
from mllib.runners.base_runners import BaseRunner
from mllib.runners.configs import BaseExperimentConfig
from mllib.tasks.base_tasks import AbstractTask

from trainers import ActivityOptimizationParams, AdversarialTrainer, AdversarialParams, ConsistentActivationModelAdversarialTrainer, MultiAttackEvaluationTrainer
from utils import write_pickle

@define(slots=False)
class AdversarialExperimentConfig(BaseExperimentConfig):
    adv_config: AdversarialParams = AdversarialParams()

@define(slots=False)
class ConsistentActivationAdversarialExperimentConfig(AdversarialExperimentConfig):
    act_opt_config: ActivityOptimizationParams = ActivityOptimizationParams()

class AdversarialExperimentRunner(BaseRunner):
    def get_experiment_dir(self, logdir, model_name, exp_name):
        def is_exp_complete(i):
            return os.path.exists(os.path.join(logdir, str(i), 'task.pkl'))
        if self.load_model_from_ckp:
            return self.ckp_dir
        exp_params = self.task.get_experiment_params()
        exp_name = f'-{exp_name}' if len(exp_name) > 0 else exp_name
        task_name = type(self.task).__name__
        dataset = self.task.get_dataset_params().dataset.name.lower()
        tr_att_p = self.task.get_experiment_params().adv_config.training_attack_params
        if tr_att_p is None:
            eps = 0.0
        else:
            eps = tr_att_p.eps
        logdir = os.path.join(exp_params.logdir, f'{dataset}-{eps}', task_name+exp_name)
        exp_num = 0
        while is_exp_complete(exp_num):
            exp_num += 1
        logdir = os.path.join(logdir, str(exp_num))
        if os.path.exists(logdir):
            shutil.rmtree(logdir)
        print(f'writing logs to {logdir}')
        return logdir
        
    def create_trainer(self):
        trainer_params = self.create_trainer_params()
        adv_trainer_params = AdversarialTrainer.get_params()
        adv_trainer_params.trainer_params = trainer_params
        adv_trainer_params.adversarial_params = self.task.get_experiment_params().adv_config
        self.trainer = adv_trainer_params.cls(adv_trainer_params)

class AdversarialAttackBatteryRunner(AdversarialExperimentRunner):
    def create_trainer(self):
        trainer_params = self.create_trainer_params()
        adv_trainer_params = MultiAttackEvaluationTrainer.get_params()
        adv_trainer_params.trainer_params = trainer_params
        adv_trainer_params.adversarial_params = self.task.get_experiment_params().adv_config
        self.trainer = adv_trainer_params.cls(adv_trainer_params)
    
    def save_task(self):
        adv_config = self.task.get_experiment_params().adv_config
        write_pickle(adv_config, os.path.join(self.trainer.logdir, 'adv_config.pkl'))

class ConsistentActivationAdversarialExperimentRunner(AdversarialExperimentRunner):
    def create_trainer(self):
        trainer_params = self.create_trainer_params()
        adv_trainer_params = ConsistentActivationModelAdversarialTrainer.get_params()
        adv_trainer_params.trainer_params = trainer_params
        adv_trainer_params.adversarial_params = self.task.get_experiment_params().adv_config
        adv_trainer_params.act_opt_params = self.task.get_experiment_params().act_opt_config
        self.trainer = adv_trainer_params.cls(adv_trainer_params)