import os
import shutil
from attrs import define, field
from mllib.runners.base_runners import BaseRunner
import torch
from mllib.datasets.dataset_factory import SupportedDatasets
from adversarialML.biologically_inspired_models.src.utils import write_pickle

class AdversarialExperimentRunner(BaseRunner):
    def create_model(self) -> torch.nn.Module:
        p = self.task.get_model_params()
        model: torch.nn.Module = p.cls(p)
        if self.load_model_from_ckp:
            src_model = self.load_model()
            # model.load_state_dict(src_model.state_dict(), strict=False)
            src_param_dict = {n:p for n,p in src_model.named_parameters()}
            for n,p in model.named_parameters():
                if (n in src_param_dict) and (src_param_dict[n].shape == p.shape):
                    print(f'loading {n} from source model')
                    p.data = src_param_dict[n].data
                else:
                    print(f'keeping {n} from model')
        return model
        
    def get_experiment_dir(self, logdir, exp_name):
        def is_exp_complete(i):
            return os.path.exists(os.path.join(logdir, str(i), 'task.pkl'))
        exp_params = self.task.get_experiment_params()
        exp_name = f'-{exp_name}' if len(exp_name) > 0 else exp_name
        task_name = type(self.task).__name__
        dataset = self.task.get_dataset_params().dataset.name.lower()
        tr_att_p = self.task.get_experiment_params().trainer_params.adversarial_params.training_attack_params
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

class AdversarialAttackBatteryRunner(AdversarialExperimentRunner):
    def get_experiment_dir(self, *args, **kwargs):
        d = os.path.dirname(os.path.dirname(self.ckp_pth))
        print(d)
        return d
    
    def save_task(self):
        if not os.path.exists(os.path.join(self.trainer.logdir, 'task.pkl')):
            super().save_task()
        adv_config = self.task.get_experiment_params().trainer_params.adversarial_params
        write_pickle(adv_config, os.path.join(self.trainer.logdir, 'adv_config.pkl'))

class RandomizedSmoothingRunner(AdversarialAttackBatteryRunner):
    def save_task(self):
        if not os.path.exists(os.path.join(self.trainer.logdir, 'task.pkl')):
            self.task.save_task(os.path.join(self.trainer.logdir, 'task.pkl'))
        randomized_smoothing_params = self.task.get_experiment_params().randomized_smoothing_config
        write_pickle(randomized_smoothing_params, os.path.join(self.trainer.logdir, 'randomized_smoothing_config.pkl'))