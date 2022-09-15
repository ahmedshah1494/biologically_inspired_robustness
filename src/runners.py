import os
import shutil
from attrs import define, field
from mllib.runners.base_runners import BaseRunner
import torch
from mllib.datasets.dataset_factory import SupportedDatasets
from adversarialML.biologically_inspired_models.src.utils import write_pickle
import webdataset as wds

class AdversarialExperimentRunner(BaseRunner):
    def create_model(self) -> torch.nn.Module:
        p = self.task.get_model_params()
        model: torch.nn.Module = p.cls(p)
        if self.load_model_from_ckp:
            src_model = self.load_model()
            src_sd = src_model.state_dict()
            tgt_sd = model.state_dict()
            mismatch = model.load_state_dict(src_sd, strict=False)
            if len(mismatch.missing_keys) > 0:
                for k in mismatch.missing_keys:
                    print(f'keeping {k} from model')
            if len(mismatch.unexpected_keys) > 0:
                print('got unexpected keys:', mismatch.unexpected_keys)
            # for k in tgt_sd:
            #     if k in src_sd:
            #         tgt_sd[k] = src_sd[k]
            #         print(f'loading {k} from source model')
            #     else:
            #         print(f'keeping {k} from model')
            # src_param_dict = {n:p for n,p in src_model.named_parameters()}
            # for n,p in model.named_parameters():
            #     if (n in src_param_dict) and (src_param_dict[n].shape == p.shape):
            #         print(f'loading {n} from source model')
            #         p.data = src_param_dict[n].data
            #     else:
            #         print(f'keeping {n} from model')
        return model
        # return self.load_model()
    
    def create_dataloaders(self):
        train_dataset, val_dataset, test_dataset = self.create_datasets()
        p = self.task.get_experiment_params()
        
        ds = self.task.get_dataset_params().dataset
        if isinstance(train_dataset, wds.WebDataset):
            num_workers = 16

            train_dataset = train_dataset.batched(p.batch_size, partial=False)
            val_dataset = val_dataset.batched(p.batch_size, partial=False)
            test_dataset = test_dataset.batched(p.batch_size, partial=False)

            train_loader = wds.WebLoader(train_dataset, batch_size=None, shuffle=False, num_workers=num_workers).with_length(len(train_dataset) // p.batch_size)
            val_loader = wds.WebLoader(val_dataset, batch_size=None, shuffle=False, num_workers=num_workers).with_length(len(val_dataset) // p.batch_size)
            test_loader = wds.WebLoader(test_dataset, batch_size=None, shuffle=False, num_workers=num_workers).with_length(len(test_dataset) // p.batch_size)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=p.batch_size, shuffle=True, num_workers=8, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=p.batch_size, shuffle=False, num_workers=8, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=p.batch_size, shuffle=False, num_workers=8, pin_memory=True)

        return train_loader, val_loader, test_loader
        
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
    def __init__(self, task, ckp_pth: str = None, load_model_from_ckp: bool = False, output_to_ckp_dir=True) -> None:
        self.output_to_ckp_dir = output_to_ckp_dir
        super().__init__(task, ckp_pth, load_model_from_ckp)

    def get_experiment_dir(self, logdir, exp_name):
        if self.output_to_ckp_dir:
            d = os.path.dirname(os.path.dirname(self.ckp_pth))
            print(d)
        else:
            def is_exp_complete(i):
                return os.path.exists(os.path.join(logdir, str(i), 'task.pkl'))
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
        randomized_smoothing_params = self.task.get_experiment_params().randomized_smoothing_config
        write_pickle(randomized_smoothing_params, os.path.join(self.trainer.logdir, 'randomized_smoothing_config.pkl'))