import os
import re
import shutil
from attrs import define, field
from mllib.runners.base_runners import BaseRunner
from mllib.runners.configs import BaseExperimentConfig
import torch
from mllib.datasets.dataset_factory import SupportedDatasets
from rblur.utils import write_pickle
import webdataset as wds
@define(slots=False)
class TransferLearningExperimentConfig(BaseExperimentConfig):
    seed_model_path: str = None
    keys_to_skip_regex: str = None
    keys_to_freeze_regex: str = None
    prefix_map: dict = {}

def print_num_params(model):
    ntrainable = 0
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            ntrainable += p.numel()
        total += p.numel()
    print(f'total parameters={total/1e6}M\ntrainable parameters={ntrainable/1e6}M')

def load_params_into_model(src_model: torch.nn.Module, tgt_model: torch.nn.Module, keys_to_skip_rgx=None, keys_to_freeze_regex=None,
                           prefix_map={}):
    if isinstance(src_model, dict):
        # This condition is used to load pytorch lightning checkpoints into
        # non-PL trainers, usually for evaluation
        src_sd = src_model['state_dict']
        # state dicts in PL checkpoint contain keys of the form "model.{...}",
        # so we must remove "model." to match them with model keys.
        src_sd = {k.replace('model.','', 1): v for k,v in src_sd.items()}
    else:
        src_sd = src_model.state_dict()
    if keys_to_skip_rgx is not None:
        src_sd = {k:v for k,v in src_sd.items() if not re.match(keys_to_skip_rgx, k)}
    new_src_sd = {}
    for k, v in src_sd.items():
        for prefix, replacement in sorted(prefix_map.items(), key=lambda e: len(e[0]), reverse=True):
            if k.startswith(prefix):
                k = replacement + k[len(prefix):]
        new_src_sd[k] = v
    src_sd = new_src_sd
    mismatch = tgt_model.load_state_dict(src_sd, strict=False)
    if len(mismatch.missing_keys) > 0:
        for k in mismatch.missing_keys:
            print(f'keeping {k} from target model')
    if len(mismatch.unexpected_keys) > 0:
        print('got unexpected keys:', mismatch.unexpected_keys)
    if keys_to_freeze_regex is not None:
        for n, p in tgt_model.named_parameters():
            if re.match(keys_to_freeze_regex, n):
                p.requires_grad = False
                print(f'freezing {n} in target model')
    return tgt_model

class AdversarialExperimentRunner(BaseRunner):
    def create_model(self) -> torch.nn.Module:
        p = self.task.get_model_params()
        model: torch.nn.Module = p.cls(p)
        ep = self.task.get_experiment_params()
        if self.load_model_from_ckp or (getattr(ep, 'seed_model_path', None) is not None) :
            if self.ckp_pth is None:
                self.ckp_pth = getattr(ep, 'seed_model_path', None)
            src_model = self.load_model()
            if self.load_model_from_ckp:
                model = load_params_into_model(src_model, model)
            else:
                print(self.ckp_pth)
                model = load_params_into_model(src_model, model, getattr(ep, 'keys_to_skip_regex', None), getattr(ep, 'keys_to_freeze_regex', None), getattr(ep, 'prefix_map', {}))
        print_num_params(model)
        return model
    
    def create_dataloaders(self):
        train_dataset, val_dataset, test_dataset = self.create_datasets()
        p = self.task.get_experiment_params()
        
        ds = self.task.get_dataset_params().dataset
        if isinstance(train_dataset, wds.WebDataset):
            num_workers = 8 // torch.cuda.device_count()

            train_dataset = train_dataset.shuffle(10_000).batched(p.batch_size, partial=False)
            val_dataset = val_dataset.batched(p.batch_size, partial=False)
            test_dataset = test_dataset.batched(p.batch_size, partial=True)

            train_loader = wds.WebLoader(train_dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=num_workers)#.with_length(len(train_dataset) // p.batch_size)
            val_loader = wds.WebLoader(val_dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=num_workers)#.with_length(len(val_dataset) // p.batch_size)
            test_loader = wds.WebLoader(test_dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=num_workers)#.with_length(len(test_dataset) // p.batch_size)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=p.batch_size, shuffle=True, num_workers=10, pin_memory=True, persistent_workers=True, drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=p.batch_size, shuffle=False, num_workers=10, pin_memory=True, persistent_workers=True, drop_last=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=p.batch_size, shuffle=False, num_workers=8)

        return train_loader, val_loader, test_loader

    def _is_exp_complete(self, logdir, i):
        return os.path.exists(os.path.join(logdir, str(i), 'metrics.json')) or os.path.exists(os.path.join(logdir, str(i), 'adv_metrics.json'))

    def _get_expdir(self, logdir, exp_name):
        exp_name = f'-{exp_name}' if len(exp_name) > 0 else exp_name
        task_name = type(self.task).__name__
        dataset = self.task.get_dataset_params().dataset.name.lower()
        tr_att_p = self.task.get_experiment_params().trainer_params.adversarial_params.training_attack_params
        if tr_att_p is None:
            eps = 0.0
        else:
            eps = tr_att_p.eps
        expdir = os.path.join(logdir, f'{dataset}-{eps}', task_name+exp_name)
        return expdir
        
    # def get_experiment_dir(self, logdir, exp_name):
    #     logdir = self._get_expdir(logdir, exp_name)
    #     exp_num = 0
    #     while self._is_exp_complete(exp_num):
    #         exp_num += 1
    #     logdir = os.path.join(logdir, str(exp_num))
    #     if os.path.exists(logdir):
    #         shutil.rmtree(logdir)
    #     os.makedirs(logdir)
    #     print(f'writing logs to {logdir}')
    #     return logdir

class AdversarialAttackBatteryRunner(AdversarialExperimentRunner):
    def __init__(self, task, num_trainings: int = 1, ckp_pth: str = None, load_model_from_ckp: bool = False, output_to_ckp_dir=True, wrap_trainer_with_lightning: bool = False, lightning_kwargs=...) -> None:
        self.output_to_ckp_dir = output_to_ckp_dir
        super().__init__(task, num_trainings, ckp_pth, load_model_from_ckp, wrap_trainer_with_lightning, lightning_kwargs)
    
    # def create_model(self) -> torch.nn.Module:
    #     p = self.task.get_model_params()
    #     model: torch.nn.Module = p.cls(p)
    #     if self.load_model_from_ckp:
    #         src_model = self.load_model()
    #         model = load_params_into_model(src_model, model)
    #     print_num_params(model)
    #     return model

    def get_experiment_dir(self, logdir, exp_name):
        if self.output_to_ckp_dir:
            d = os.path.dirname(os.path.dirname(self.ckp_pth))
            print(d)
        else:
            def is_exp_complete(i):
                return os.path.exists(os.path.join(logdir, str(i), 'metrics.json')) or os.path.exists(os.path.join(logdir, str(i), 'adv_metrics.json'))
            exp_params = self.task.get_experiment_params()
            exp_name = f'-{exp_name}' if len(exp_name) > 0 else exp_name
            task_name = self.task._cls.__name__
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
        randomized_smoothing_params = self.task.get_experiment_params().trainer_params.randomized_smoothing_params
        write_pickle(randomized_smoothing_params, os.path.join(self.trainer.logdir, 'randomized_smoothing_config.pkl'))