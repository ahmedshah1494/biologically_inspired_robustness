from enum import Enum, auto
import os
from typing import List, Type, Union, Tuple
from attrs import define, field
from mllib.trainers.base_trainers import Trainer as _Trainer
from mllib.utils.metric_utils import compute_accuracy, get_preds_from_logits
from mllib.param import BaseParameters
from numpy import iterable
import torch
import numpy as np
import torchattacks
from einops import rearrange

from mllib.adversarial.attacks import AbstractAttackConfig, FoolboxAttackWrapper, FoolboxCWL2AttackWrapper
from adversarialML.biologically_inspired_models.src.models import ConsistencyOptimizationMixin
from adversarialML.biologically_inspired_models.src.pruning import PruningMixin
from adversarialML.biologically_inspired_models.src.utils import aggregate_dicts, merge_iterables_in_dict, write_json, write_pickle, load_json, recursive_dict_update, load_pickle

@define(slots=False)
class AdversarialParams:
    training_attack_params: AbstractAttackConfig = None
    testing_attack_params: List[AbstractAttackConfig] = [None]

class AdversarialTrainer(_Trainer, PruningMixin):    
    @define(slots=False)    
    class AdversarialTrainerParams(BaseParameters):
        trainer_params: Type[_Trainer.TrainerParams] = field(factory=lambda: _Trainer.TrainerParams(None))
        adversarial_params: Type[AdversarialParams] = field(factory=AdversarialParams)

    @classmethod
    def get_params(cls):
        return cls.AdversarialTrainerParams(cls)

    def __init__(self, params: AdversarialTrainerParams):
        super().__init__(params.trainer_params)
        print(self.model)
        self.params = params
        self.training_adv_attack = self._maybe_get_attacks(params.adversarial_params.training_attack_params)
        if isinstance(self.training_adv_attack, tuple):
            self.training_adv_attack = self.training_adv_attack[1]
        self.testing_adv_attacks = self._maybe_get_attacks(params.adversarial_params.testing_attack_params)
        self.data_and_pred_filename = 'data_and_preds.pkl'
        self.metrics_filename = 'metrics.json'

    def _get_attack_from_params(self, p: Union[AbstractAttackConfig, Tuple[str, AbstractAttackConfig]]):
        if isinstance(p, tuple):
            name, p = p
        else:
            name = None
        if p is not None:
            if p.model is None:
                p.model = self.model.eval()
            return name, p._cls(p.model, **(p.asdict()))

    def _maybe_get_attacks(self, attack_params: Union[AbstractAttackConfig, List[AbstractAttackConfig]]):
        if attack_params is None:
            attack = None
        else:
            if iterable(attack_params):
                attack = [self._get_attack_from_params(p) for p in attack_params]
            else:
                attack = self._get_attack_from_params(attack_params)
        return attack
    
    def _maybe_attack_batch(self, batch, adv_attack):
        x,y = batch
        if adv_attack is not None:
            if x.dim() == 5:
                y_ = torch.cat([y]*x.shape[1], 0)
                x = rearrange(x, 'b n c h w -> (b n) c h w')
                x = adv_attack(x, y_)
                x = rearrange(x, '(b n) c h w -> b n c h w', b = len(y))
            else:
                x = adv_attack(x, y)
        return x,y

    def train_step(self, batch, batch_idx):
        batch = self._maybe_attack_batch(batch, self.training_adv_attack)
        return super().train_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        test_eps = [(p.eps if p is not None else 0.) for p in self.params.adversarial_params.testing_attack_params]
        test_pred = {}
        adv_x = {}
        test_loss = {}
        test_acc = {}
        test_logits = {}
        for name, atk in self.testing_adv_attacks:
            if isinstance(atk, FoolboxAttackWrapper):
                eps = atk.run_kwargs.get('epsilons', [float('inf')])[0]
            elif isinstance(atk, torchattacks.attack.Attack):
                eps = atk.eps
            else:
                raise NotImplementedError(f'{type(atk)} is not supported')
            x,y = self._maybe_attack_batch(batch, atk)

            logits, loss = self._get_outputs_and_loss(x, y)
            logits = logits.detach().cpu()
            
            y = y.detach().cpu()
            acc, _ = compute_accuracy(logits, y)

            preds = get_preds_from_logits(logits)
            loss = loss.mean().detach().cpu()

            test_pred[eps] = preds.numpy().tolist()
            adv_x[eps] = x.detach().cpu().numpy()
            test_loss[eps] = loss
            test_acc[eps] = acc
            test_logits[eps] = logits.numpy()
        metrics = {f'test_acc_{k}':v for k,v in test_acc.items()}
        return {'preds':test_pred, 'labels':y.numpy().tolist(), 'inputs': adv_x, 'logits':test_logits}, metrics
    
    def test_epoch_end(self, outputs, metrics):
        outputs = aggregate_dicts(outputs)
        test_eps = [(p.eps if p is not None else 0.) for p in self.params.adversarial_params.testing_attack_params]
        new_outputs = aggregate_dicts(outputs)
        new_outputs = merge_iterables_in_dict(new_outputs)
        
        test_acc = {}
        for i,eps in enumerate(test_eps):
            acc = (np.array(new_outputs['preds'][eps]) == np.array(new_outputs['labels'])).astype(float).mean()
            test_acc[eps] = acc
        new_outputs['test_acc'] = test_acc
        return new_outputs, metrics
    
    def train(self):
        metrics = super().train()
        val_acc = metrics['val_accuracy']
        # if val_acc > 0.12:
        #     self.prune()

    def prune(self):
        self.iterative_pruning_wrapper(0, self.l1_unstructured_pruning_with_retraining, 0.1)
    
    def save_training_logs(self, train_acc, test_accs):
        metrics = {
            'train_acc':train_acc,
            'test_accs':test_accs,
        }
        write_json(metrics, os.path.join(self.logdir, self.metrics_filename))

    def save_data_and_preds(self, preds, labels, inputs, logits):
        d = {}
        for k in preds.keys():
            d[k] = {
                'X': inputs[k],
                'Y': labels,
                'Y_pred': preds[k],
                'logits': logits[k]
            }
        write_pickle(d, os.path.join(self.logdir, self.data_and_pred_filename))
    
    def save_logs_after_test(self, train_metrics, test_outputs):
        self.save_training_logs(train_metrics['train_accuracy'], test_outputs['test_acc'])
        self.save_data_and_preds(test_outputs['preds'], test_outputs['labels'], test_outputs['inputs'], test_outputs['logits'])

    def test(self):
        test_outputs, test_metrics = self.test_loop(post_loop_fn=self.test_epoch_end)
        print('test metrics:')
        print(test_metrics)
        _, train_metrics = self._batch_loop(self.val_step, self.train_loader, 0, logging=False)
        for k in train_metrics:
            train_metrics[k.replace('val', 'train')] = train_metrics.pop(k)
        train_metrics = aggregate_dicts(train_metrics)
        self.save_logs_after_test(train_metrics, test_outputs)
        
    def _log(self, logs, step):
        for k,v in logs.items():
            if isinstance(v, dict):
                self.logger.add_scalars(k, v, global_step=step)
            elif not iterable(v):
                self.logger.add_scalar(k, v, global_step=step)

class ActivityOptimizationSchedule(Enum):
    CONST = auto()
    GEOM = auto()
    LINEAR = auto()

@define(slots=False)
class ActivityOptimizationParams:
    act_opt_lr_warmup_schedule: Type[ActivityOptimizationSchedule] = ActivityOptimizationSchedule.CONST
    init_act_opt_lr: float = 1e-2
    num_warmup_epochs: int = 5

class ConsistentActivationModelAdversarialTrainer(AdversarialTrainer):
    class ConsistentActivationModelAdversarialTrainerParams(BaseParameters):
        def __init__(self, cls):
            super().__init__(cls)
            self.trainer_params: Type[_Trainer.TrainerParams] = _Trainer.TrainerParams(None)
            self.adversarial_params: Type[AdversarialParams] = AdversarialParams
            self.act_opt_params: ActivityOptimizationParams = ActivityOptimizationParams()
    
    @classmethod
    def get_params(cls):
        return cls.ConsistentActivationModelAdversarialTrainerParams(cls)

    def __init__(self, params: ConsistentActivationModelAdversarialTrainerParams):
        super().__init__(params)
        self.params = params
        self._load_max_act_opt_lrs()

    def _load_max_act_opt_lrs(self):
        self.max_act_opt_lrs = {}
        for n, m in self.model.named_modules():
            if isinstance(m, ConsistencyOptimizationMixin):
                self.max_act_opt_lrs[n] = m.act_opt_step_size

    def _update_act_opt_lrs(self, epoch_idx: int):
        if (epoch_idx < self.params.act_opt_params.num_warmup_epochs) and self.params.act_opt_params.act_opt_lr_warmup_schedule != ActivityOptimizationSchedule.CONST:
            for n,m in self.model.named_modules():
                if n in self.max_act_opt_lrs:
                    init_lr = min(self.params.act_opt_params.init_act_opt_lr, self.max_act_opt_lrs[n])
                    if self.params.act_opt_params.act_opt_lr_warmup_schedule == ActivityOptimizationSchedule.GEOM:
                        m.act_opt_step_size = np.geomspace(init_lr, 
                                                            self.max_act_opt_lrs[n], 
                                                            self.params.act_opt_params.num_warmup_epochs)[epoch_idx]
                    if self.params.act_opt_params.act_opt_lr_warmup_schedule == ActivityOptimizationSchedule.LINEAR:
                        m.act_opt_step_size = np.linspace(init_lr, 
                                                            self.max_act_opt_lrs[n], 
                                                            self.params.act_opt_params.num_warmup_epochs)[epoch_idx]
                    print(n, m.act_opt_step_size)

    def train_loop(self, epoch_idx, post_loop_fn=None):
        self._update_act_opt_lrs(epoch_idx)
        return super().train_loop(epoch_idx, post_loop_fn)

def compute_adversarial_success_rate(clean_preds, preds, labels, target_labels):
    if (labels == target_labels).all():
        adv_succ = ((clean_preds == labels) & (preds != labels)).astype(float).mean()
    else:
        adv_succ = (preds == target_labels).astype(float).mean()
    return adv_succ

class MultiAttackEvaluationTrainer(AdversarialTrainer):
    def __init__(self, params):
        super().__init__(params)
        self.metrics_filename = 'adv_metrics.json'
        self.data_and_pred_filename = 'adv_data_and_preds.pkl'

    def save_training_logs(self, train_acc, test_accs):
        outfile = os.path.join(self.logdir, self.metrics_filename)
        if os.path.exists(outfile):
            self.metrics_filename = '.'+self.metrics_filename
            super().save_training_logs(train_acc, test_accs)
            metrics = load_json(os.path.join(self.logdir, self.metrics_filename))
            self.metrics_filename = self.metrics_filename[1:]
            old_metrics = load_json(outfile)
            recursive_dict_update(old_metrics, metrics)
            write_json(metrics, outfile)
        else:
            super().save_training_logs(train_acc, test_accs)
    
    def save_data_and_preds(self, preds, labels, inputs, logits):
        outfile = os.path.join(self.logdir, self.data_and_pred_filename)
        if os.path.exists(outfile):
            self.data_and_pred_filename = '.'+self.data_and_pred_filename
            super().save_data_and_preds(preds, labels, inputs, logits)
            metrics = load_pickle(os.path.join(self.logdir, self.data_and_pred_filename))
            self.data_and_pred_filename = self.data_and_pred_filename[1:]
            old_metrics = load_pickle(outfile)
            recursive_dict_update(old_metrics, metrics)
            write_pickle(metrics, outfile)
        else:
            super().save_data_and_preds(preds, labels, inputs, logits)

    def test_epoch_end(self, outputs, metrics):
        outputs = aggregate_dicts(outputs)
        new_outputs = aggregate_dicts(outputs)
        new_outputs = merge_iterables_in_dict(new_outputs)
        
        labels = np.array(new_outputs['labels'])
        test_acc = {}
        adv_succ = {}
        for k in new_outputs['preds'].keys():
            target_labels = np.array(new_outputs['target_labels'][k])
            preds = np.array(new_outputs['preds'][k])
            clean_preds = np.array(new_outputs['preds'][sorted(new_outputs['preds'].keys())[0]])
            acc = (preds == labels).astype(float).mean()
            test_acc[k] = acc
            adv_succ[k] = compute_adversarial_success_rate(clean_preds, preds, labels, target_labels)
        new_outputs['test_acc'] = test_acc
        new_outputs['adv_succ'] = adv_succ
        write_json(adv_succ, os.path.join(self.logdir, 'adv_succ.json'))
        return new_outputs, metrics

    def test_step(self, batch, batch_idx):
        test_pred = {}
        adv_x = {}
        test_loss = {}
        test_acc = {}
        test_logits = {}
        target_labels = {}
        for name, atk in self.testing_adv_attacks:
            if isinstance(atk, FoolboxCWL2AttackWrapper):
                eps = atk.attack.confidence
            elif isinstance(atk, FoolboxAttackWrapper):
                eps = atk.run_kwargs.get('epsilons', [float('inf')])[0]
            elif isinstance(atk, torchattacks.attack.Attack):
                eps = atk.eps
            else:
                raise NotImplementedError(f'{type(atk)} is not supported')
            atk_name = f"{atk.__class__.__name__ if name is None else name}-{eps}"

            x,y = self._maybe_attack_batch(batch, atk if eps > 0 else None)

            logits, loss = self._get_outputs_and_loss(x, y)
            logits = logits.detach().cpu()
            
            y = y.detach().cpu()
            acc, _ = compute_accuracy(logits, y)
            if isinstance(atk, torchattacks.attack.Attack) and atk._targeted:
                y_tgt = atk._get_target_label(*batch)
            else:
                y_tgt = y

            preds = get_preds_from_logits(logits)
            loss = loss.mean().detach().cpu()

            test_pred[atk_name] = preds.numpy().tolist()
            adv_x[atk_name] = x.detach().cpu().numpy()
            test_loss[atk_name] = loss
            test_acc[atk_name] = acc
            test_logits[atk_name] = logits.numpy()
            target_labels[atk_name] = y_tgt.detach().cpu().numpy().tolist()
        metrics = {f'test_acc_{k}':v for k,v in test_acc.items()}
        return {'preds':test_pred, 'labels':y.numpy().tolist(), 'inputs': adv_x, 'target_labels':target_labels, 'logits': test_logits}, metrics