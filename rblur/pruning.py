from mllib.trainers.base_trainers import Trainer

from copy import deepcopy
import torch

import torch.nn.utils.prune as prune

class PruningMixin:
    def iterative_pruning_wrapper(self: Trainer, fine_tune_epochs, prune_step_fn, *args, **kwargs):
        def _get_train_acc():
            with torch.no_grad():
                _, metrics = self._batch_loop(self.train_step, self.train_loader, i, logging=False)
            return metrics['train_accuracy']
        
        def count_nz_params():
            c = 0
            for module in self.model.modules():
                for name in ['weight', 'bias']:
                    if hasattr(module, name) and getattr(module, name) is not None:
                        W = getattr(module, name).data
                        nz = ((W != 0) & (~torch.isnan(W))).int().sum().cpu().detach().numpy()
                        c += nz
            return c

        model = self.model
        orig_model = deepcopy(model)
        i = 0
        new_acc = old_acc = _get_train_acc()
        max_pruning_iters = sum([p.numel() for p in model.parameters()])
        nz_params = count_nz_params()
        tol = 0.01
        while (i == 0 or ((old_acc - new_acc < tol) and (i < max_pruning_iters)) and (nz_params > 0)):
            print(nz_params, old_acc-new_acc)
            old_sd = deepcopy(model.state_dict())
            prune_step_fn(*args, **kwargs)
            old_acc = new_acc
            new_acc =_get_train_acc()
            
            j = 0
            while (old_acc - new_acc >= tol) and j < fine_tune_epochs:
                self.train_loop(i, self.train_epoch_end)
                new_acc = _get_train_acc()
                j += 1
            nz_params = count_nz_params()
            i+=1
        if (old_acc - new_acc > tol):
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