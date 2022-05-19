from mllib.trainers.base_trainers import Trainer

from copy import deepcopy
import torch

import torch.nn.utils.prune as prune

class PruningMixin:
    def iterative_pruning_wrapper(self: Trainer, fine_tune_epochs, prune_step_fn, *args, **kwargs):
        model = self.model
        orig_model = deepcopy(model)
        i = 0
        _, metrics = self._batch_loop(self.train_step, self.train_loader, i, logging=False)
        new_acc = old_acc = metrics['train_accuracy']
        max_pruning_iters = sum([p.numel() for p in model.parameters()])
        tol = 0.01
        while (i == 0 or ((old_acc - new_acc < tol) and (i < max_pruning_iters))):
            old_sd = deepcopy(model.state_dict())
            prune_step_fn(*args, **kwargs)
            old_acc = new_acc
            _, metrics = self._batch_loop(self.train_step, self.train_loader, i, logging=False)
            new_acc = metrics['train_accuracy']
            
            j = 0
            while (old_acc - new_acc >= tol) and j < fine_tune_epochs:
                self.train_loop(i, self.train_epoch_end)
                _, metrics = self._batch_loop(self.train_step, self.train_loader, i, logging=False)
                new_acc = metrics['train_accuracy']
                j += 1
            i+=1
        if (old_acc - new_acc > 0.01):
            if i == 1:
                model = orig_model
            else:
                model.load_state_dict(old_sd)
        self.remove_pruning_mask()
        return model

    def remove_pruning_mask(self):
        def _remove_mask(module, name):
            if hasattr(module, name) and prune.is_pruned(module):
                prune.remove(module, name)
        for module in self.model.modules():
            _remove_mask(module, 'weight')
            _remove_mask(module, 'bias')

    def l1_unstructured_pruning_with_retraining(self, pruning_factor):
        def _prune(module, name):
            if hasattr(module, name) and getattr(module, name) is not None:
                W = getattr(module, name).data
                nz = ((W != 0) & (~torch.isnan(W))).int().sum().cpu().detach().numpy()
                if nz > 0:
                    prune.l1_unstructured(module, name, int(max(1, nz * pruning_factor)))
        for module in self.model.modules():
            _prune(module, 'weight')
            _prune(module, 'bias')