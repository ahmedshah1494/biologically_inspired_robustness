import os
from typing import List, Type, Union
from attrs import define
from mllib.trainers.base_trainers import Trainer as _Trainer
from mllib.utils.metric_utils import compute_accuracy, get_preds_from_logits
from mllib.param import BaseParameters
from numpy import iterable
import torch
import numpy as np

from models import ConsistentActivationClassifier
from mllib.adversarial.attacks import AbstractAttackConfig, AttackParamFactory
from pruning import PruningMixin
from trainer_utils import _make_adv_datset, make_dataloader
from utils import aggregate_dicts, merge_iterables_in_dict, write_json, write_pickle

@define(slots=False)
class AdversarialParams:
    training_attack_params: AbstractAttackConfig = None
    testing_attack_params: List[AbstractAttackConfig] = [None]

class AdversarialTrainer(_Trainer, PruningMixin):
    @define(slots=False)
    class AdversarialTrainerParams(BaseParameters):
        trainer_params: Type[_Trainer.TrainerParams] = _Trainer.TrainerParams(None)
        adversarial_params: Type[AdversarialParams] = AdversarialParams

    @classmethod
    def get_params(cls):
        return cls.AdversarialTrainerParams(cls)

    def __init__(self, params: AdversarialTrainerParams):
        super().__init__(params.trainer_params)
        self.params = params
        self.training_adv_attack = self._maybe_get_attacks(params.adversarial_params.training_attack_params)

        self.testing_adv_attacks = self._maybe_get_attacks(params.adversarial_params.testing_attack_params)

    def _get_attack_from_params(self, p: AbstractAttackConfig):
        return p._cls(self.model, **(p.asdict()))

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
            x = adv_attack(x, y)
        return x,y

    def train_step(self, batch, batch_idx):
        batch = self._maybe_attack_batch(batch, self.training_adv_attack)
        return super().train_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        test_eps = [p.eps for p in self.params.adversarial_params.testing_attack_params]
        test_pred = {}
        adv_x = {}
        test_loss = {}
        test_acc = {}
        for eps, atk in zip(test_eps, self.testing_adv_attacks):
            x,y = self._maybe_attack_batch(batch, atk)
            y = y.detach().cpu()

            logits, loss = self._get_outputs_and_loss(x, y)
            logits = logits.detach().cpu()

            acc, _ = compute_accuracy(logits, y)

            preds = get_preds_from_logits(logits)
            loss = loss.mean().detach().cpu()

            test_pred[eps] = preds.numpy().tolist()
            adv_x[eps] = x.detach().cpu().numpy()
            test_loss[eps] = loss
            test_acc[eps] = acc
        metrics = {f'test_acc_{k}':v for k,v in test_acc.items()}
        return {'preds':test_pred, 'labels':y.numpy().tolist(), 'inputs': adv_x}, metrics
    
    def test_epoch_end(self, outputs, metrics):
        outputs = aggregate_dicts(outputs)
        test_eps = [p.eps for p in self.params.adversarial_params.testing_attack_params]
        new_outputs = aggregate_dicts(outputs)
        new_outputs = merge_iterables_in_dict(new_outputs)
        
        test_acc = {}
        for i,eps in enumerate(test_eps):
            acc = (np.array(new_outputs['preds'][eps]) == np.array(new_outputs['labels'])).astype(float).mean()
            test_acc[eps] = acc
        new_outputs['test_acc'] = test_acc
        return new_outputs, metrics
    
    def train(self):
        super().train()
        self.iterative_pruning_wrapper(0, self.l1_unstructured_pruning_with_retraining, 0.1)
    
    def save_training_logs(self, train_acc, test_accs):
        metrics = {
            'train_acc':train_acc,
            'test_accs':test_accs,
        }
        write_json(metrics, os.path.join(self.logdir, 'metrics.json'))

    def save_data_and_preds(self, preds, labels, inputs):
        d = {}
        for k in preds.keys():
            d[k] = {
                'X': inputs[k],
                'Y': labels,
                'Y_pred': preds[k]
            }
        write_pickle(d, os.path.join(self.logdir, f'data_and_preds.pkl'))

    def test(self):
        test_outputs, test_metrics = self.test_loop(post_loop_fn=self.test_epoch_end)
        print('test metrics:')
        print(test_metrics)

        _, train_metrics = self._batch_loop(self.train_step, self.train_loader, 0, logging=False)
        train_metrics = aggregate_dicts(train_metrics)
        # test_metrics = aggregate_dicts(test_metrics)
        # test_accs = {float(k.split('_')[-1]): v for k,v in test_metrics.items() if k.startswith('test_acc_')}

        self.save_training_logs(train_metrics['train_accuracy'], test_outputs['test_acc'])
        self.save_data_and_preds(test_outputs['preds'], test_outputs['labels'], test_outputs['inputs'])

    def _log(self, logs, step):
        for k,v in logs.items():
            if isinstance(v, dict):
                self.logger.add_scalars(k, v, global_step=step)
            elif not iterable(v):
                self.logger.add_scalar(k, v, global_step=step)