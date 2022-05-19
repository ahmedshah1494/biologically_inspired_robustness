import traceback
from typing import Iterable, List, Type, Union
from attr import define
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
import numpy as np

from torch import nn
import torch

from model_utils import mm, str_to_act_and_dact_fn

@define(slots=False)
class CommonModelParams:
    input_size: Union[int, List[int]] = 0
    num_units: Union[int, List[int]] = 0
    activation: Type[nn.Module] = nn.ReLU
    bias: bool = True

@define(slots=False)
class ClassificationModelParams:
    num_classes: int = 1

@define(slots=False)
class ConsistencyOptimizationParams:
    lateral_dependence_type: str ='Linear'
    act_opt_step_size: float = 0.07
    max_act_update_norm: float = 10.
    normalize_lateral_dependence: bool = False
    input_act_consistency: bool = False
    act_act_consistency: bool = True
    backward_dependence_type:str = 'Linear'
    no_act_after_update: bool = False

    max_train_time_steps: int = 1
    max_test_time_steps: int = 1

    legacy_act_update_normalization: bool = False
    sparsify_act: bool = False
    sparsity_coeff: float = 1.

class CommonModelMixin(object):
    def load_common_params(self) -> None:
        input_size = self.params.common_params.input_size
        num_units = self.params.common_params.num_units
        activation = self.params.common_params.activation
        bias = self.params.common_params.bias

        self.input_size = input_size
        self.num_units = num_units
        self.activation = activation
        self.use_bias = bias

class ClassificationModelMixin(object):
    def load_classification_params(self) -> None:
        self.num_classes = self.params.classification_params.num_classes
        self.num_logits = (1 if self.num_classes == 2 else self.num_classes)


class ConsistencyOptimizationMixin(object):
    def load_consistency_opt_params(self) -> None:
        lateral_dependence_type = self.params.consistency_optimization_params.lateral_dependence_type
        act_opt_step_size = self.params.consistency_optimization_params.act_opt_step_size
        max_act_update_norm = self.params.consistency_optimization_params.max_act_update_norm
        normalize_lateral_dependence = self.params.consistency_optimization_params.normalize_lateral_dependence
        input_act_consistency = self.params.consistency_optimization_params.input_act_consistency
        act_act_consistency = self.params.consistency_optimization_params.act_act_consistency
        backward_dependence_type = self.params.consistency_optimization_params.backward_dependence_type
        no_act_after_update = self.params.consistency_optimization_params.no_act_after_update
        legacy_act_update_normalization = self.params.consistency_optimization_params.legacy_act_update_normalization
        sparsify_act = self.params.consistency_optimization_params.sparsify_act
        sparsity_coeff = self.params.consistency_optimization_params.sparsity_coeff
        max_train_time_steps = self.params.consistency_optimization_params.max_train_time_steps
        max_test_time_steps = self.params.consistency_optimization_params.max_test_time_steps

        self.input_act_consistency = input_act_consistency
        self.act_act_consistency = act_act_consistency
        if not (input_act_consistency or act_act_consistency):
            max_test_time_steps = max_train_time_steps = 0
        self.max_train_time_steps = max_train_time_steps
        self.max_test_time_steps = max_test_time_steps
        self.lateral_dependence_type = lateral_dependence_type
        self.backward_dependence_type = backward_dependence_type
        self.act_opt_step_size = act_opt_step_size
        self.max_act_update_norm = max_act_update_norm
        self.normalize_lateral_dependence = normalize_lateral_dependence
        self.no_act_after_update = no_act_after_update
        self.legacy_act_update_normalization = legacy_act_update_normalization
        self.sparsify_act = sparsify_act
        self.sparsity_coeff = sparsity_coeff
        self.use_pt_optimizer = False
        self.truncate_act_opt_grad = False
        
    def _get_activation_update_manual(self, act:torch.Tensor, W:torch.Tensor, b:torch.Tensor, normalize_by_act_norm=False):
        diag_mask = 1-torch.eye(W.shape[1], device=W.device)
        if len(W.shape) == 3:
            diag_mask = diag_mask.unsqueeze(0)
        W = W * diag_mask
        act_fn, dact_fn = str_to_act_and_dact_fn(self.lateral_dependence_type)
        linear_pred = mm(W, act) + b
        pred_act = act_fn(linear_pred)
        dpred_act = dact_fn(linear_pred)
        diff = act - pred_act
        act_update = diff - mm(torch.transpose(W, -2, -1), dpred_act * diff)
        diff = (act - pred_act)
        diff_sqnorm = (diff**2).sum(0)
        unnorm_loss = (0.5 * diff_sqnorm)
        if normalize_by_act_norm:
            act_sqnorm = (act ** 2).sum(0, keepdim=True) + 1e-8
            act_update = act_update / act_sqnorm - (2 * unnorm_loss.reshape(1,-1) * act) / (act_sqnorm**2)
            loss = unnorm_loss / act_sqnorm.squeeze()
        else:
            loss = unnorm_loss
        loss = loss.mean()
        if self.legacy_act_update_normalization:
            act_update /= act.shape[1]
        return act_update, loss

    def _get_activation_backward_update_manual(self, x: torch.Tensor, act: torch.Tensor, W: torch.Tensor, b: torch.Tensor, normalize_by_act_norm=False):
        act_fn, dact_fn = str_to_act_and_dact_fn(self.backward_dependence_type)
        linear_pred = mm(W, act) + b
        pred_act = act_fn(linear_pred)
        dpred_act = dact_fn(linear_pred)
        diff = x - pred_act
        act_update = -mm(W.T, dpred_act * diff)
        diff_sqnorm = (diff**2).sum(0)
        unnorm_loss = (0.5 * diff_sqnorm)
        if normalize_by_act_norm:
            act_sqnorm = (act ** 2).sum(0, keepdim=True) + 1e-8
            act_update = act_update / act_sqnorm - (2 * unnorm_loss.reshape(1,-1) * act) / (act_sqnorm**2)
            loss = unnorm_loss / act_sqnorm.squeeze()
        else:
            loss = unnorm_loss
        loss = loss.mean()
        if not torch.isfinite(loss):
            print(torch.norm(W), torch.norm(b), torch.norm(x, dim=0).mean(), torch.norm(pred_act, dim=0))
        if self.legacy_act_update_normalization:
            act_update /= act.shape[1]
        return act_update, loss

    def _get_sparsity_update(self, act:torch.Tensor):
        if self.sparsify_act:
            activation_str = self.activation.__str__()
            activation_str = activation_str[:activation_str.index("(")]
            act_fn, dact_fn = str_to_act_and_dact_fn(activation_str)
            return self.sparsity_coeff * torch.sign(act_fn(act)) * dact_fn(act)
        else:
            return 0
            
    def _optimization_step(self, step_idx, state, x, W_lat, b_lat, W_back, b_back):
        if (step_idx == 0) or (not self.no_act_after_update):
            act = self.activation(state)
        else:
            act = state
        lat_act_update = 0
        back_act_update = 0
        lat_loss = 0
        back_loss = 0
        if self.act_act_consistency:
            lat_act_update, lat_loss = self._get_activation_update_manual(act, W_lat, b_lat, normalize_by_act_norm=self.normalize_lateral_dependence)
            # lat_act_update_norm = lat_act_update.reshape(-1,lat_act_update.shape[-1]).abs().max(0)[0]
            # print(i, 'lat_loss =', lat_loss.detach().cpu().numpy().mean(), f'max update norm = {lat_act_update_norm.max():.3e}', lat_act_update.shape)
        if self.input_act_consistency:
            back_act_update, back_loss = self._get_activation_backward_update_manual(x, act, W_back, b_back)
            if not (torch.isfinite(lat_act_update).all() and torch.isfinite(back_act_update).all()):
                print(torch.norm(lat_act_update, p=2, dim=0).mean(), torch.norm(back_act_update, p=2, dim=0).mean())
            # back_act_update_norm = back_act_update.reshape(-1,back_act_update.shape[-1]).abs().max(0)[0]
            # print(f'{i} max lat update norm = {lat_act_update_norm.max():.3e} max_back_update_norm={back_act_update_norm.max()}')
        act_update_m = lat_act_update + 0.3*back_act_update
        loss = lat_loss + back_loss

        with torch.no_grad():
            if self.legacy_act_update_normalization:
                act_update_norm = torch.norm(act_update_m, p=2, dim=0)
            else:
                act_update_norm = act_update_m.reshape(-1,act_update_m.shape[-1]).abs().max(0)[0]
            act_update_norm[act_update_norm == 0] = 1e-8
            scale = self.max_act_update_norm / act_update_norm
            scale = torch.min(scale, torch.ones_like(scale))
        act_update_m = act_update_m * scale
        state = act - self.act_opt_step_size * act_update_m
        sparsity_update = self._get_sparsity_update(state)
        state = state - self.act_opt_step_size * sparsity_update
        # print(step_idx, 'loss =', loss.detach().cpu().numpy().mean(), f'max update norm = {act_update_norm.max():.3e} (/{scale.min():.3f})', act_update_m.shape, scale.shape, (state>0).float().mean().cpu().detach().numpy())
        return state, loss, act_update_m

    def _optimize_activations(self, state_hist, x, W_lat, b_lat, W_back, b_back):
        max_time_steps = self.max_train_time_steps if self.training else self.max_test_time_steps
        i = 0
        prev_loss = torch.tensor(np.inf, device=state_hist[0].device)
        best_loss = torch.tensor(np.inf, device=state_hist[0].device)
        num_bad_steps = 0
        max_num_bad_steps = 5
        if self.truncate_act_opt_grad and self.training and (max_time_steps > 0):
            truncation_idx = np.random.choice(range(max_time_steps), p=[1/max_time_steps]*max_time_steps)
        if self.use_pt_optimizer:
            dummy_var = torch.nn.Parameter(torch.zeros_like(state_hist[-1]), True)
            optimizer = torch.optim.SGD([dummy_var], lr=self.act_opt_step_size, momentum=0.9, nesterov=True)
        while (i < max_time_steps) or ((max_time_steps == -1) and (num_bad_steps < max_num_bad_steps)):
            state = state_hist[-1]
            new_state, loss, update = self._optimization_step(i, state, x, W_lat, b_lat, W_back, b_back)            
            if (loss < best_loss) and not torch.isclose(loss, prev_loss, rtol=0.001, atol=1e-4):
                best_loss = loss
                num_bad_steps = 0
            else:
                num_bad_steps += 1

            if self.truncate_act_opt_grad and self.training and (i < truncation_idx):
                new_state = new_state.detach()
            state_hist.append(new_state)
            i += 1
            prev_loss = loss

    
class ConsistentActivationLayer(AbstractModel, CommonModelMixin, ConsistencyOptimizationMixin):
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = CommonModelParams()
        classification_params: ClassificationModelParams = ClassificationModelParams()
        consistency_optimization_params: ConsistencyOptimizationParams = ConsistencyOptimizationParams()        

    def __init__(self, params: ModelParams) -> None:
        print('in ConsistentActivationLayer')
        super().__init__(params)
        self.load_common_params()
        self.load_consistency_opt_params()
        self._make_network()
        self._make_name()

    def _make_name(self):
        self.name = f'FC-{self.num_units}-{self.max_train_time_steps}-{self.max_test_time_steps}steps'
        activation_str = self.activation.__str__()
        self.name +=f'-{activation_str[:activation_str.index("(")]}'

        if not self.use_bias:
            self.name += '-noBias'

        if self.sparsify_act:
            self.name += f'-{self.sparsity_coeff:.3f}Sparse'

        ld_type = ('Normalized' if self.normalize_lateral_dependence else '')+self.lateral_dependence_type
        _inc_input_str = f'{self.backward_dependence_type}BackDep' if self.input_act_consistency else ''
        self.name = f'Max{ld_type}LatDep{_inc_input_str}{self.name}-E2E'

        if self.no_act_after_update:
            self.name += '-noActAfterUpdate'

        if self.truncate_act_opt_grad:
            self.name += '-TruncActOpt'
        if self.use_pt_optimizer:
            self.name += '-PTOptimActOpt'

    def _make_network(self):
        state_size = self.num_units + self.input_size
        self.activation = self.activation()
        self.weight = nn.Parameter(torch.empty((self.num_units, state_size)), True)
        self.bias = nn.Parameter(torch.empty((self.num_units,)).zero_(), self.use_bias)
        if self.input_act_consistency:
            self.weight_back = nn.Parameter(torch.empty((self.input_size, self.num_units)), True)
            self.bias_back = nn.Parameter(torch.empty((self.input_size,)).zero_(), self.use_bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.input_act_consistency:
            nn.init.kaiming_uniform_(self.weight_back, a=np.sqrt(5))
        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            if self.input_act_consistency:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_back)
                bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias_back, -bound, bound)

    def _input_to_preact(self, x):
        W_ff = self.weight[:, :self.input_size]
        return mm(W_ff, x) + self.bias.reshape(-1,1)

    def forward(self, x, return_state_hist=False):
        x = x.reshape(x.shape[0], -1)
        W = self.weight
        W_ff = W[:, :self.input_size]
        W_lat = W[:, self.input_size:]
        b = self.bias.reshape(-1,1)
        fwd = self._input_to_preact(x.T)
        state_hist = [fwd]
        # print(0, state_hist[-1][0])
        if self.input_act_consistency:
            W_back, bias_back = self.weight_back, self.bias_back.reshape(-1, 1)
        else:
            W_back, bias_back = None, None
        self._optimize_activations(state_hist, x.T, W_lat, b, W_back, bias_back)
        logits = state_hist[-1]
        if return_state_hist:
            state_hist = torch.stack(state_hist, dim=1).transpose(0, 2)
            return logits, state_hist
        else:
            return logits

class ConsistentActivationClassifier(ConsistentActivationLayer, ClassificationModelMixin):
    class ModelParams(ConsistentActivationLayer.ModelParams):
        classification_params: ClassificationModelParams = ClassificationModelParams()   

    def __init__(self, params: ModelParams) -> None:
        print('in ConsistentActivationClassifier')
        super().__init__(params)
        self.load_classification_params()

    def forward(self, x, return_state_hist=False):
        logits, state_hist = super().forward(x, return_state_hist=True)
        num_logits = (1 if self.num_classes == 2 else self.num_classes)
        logits = logits.T[:, :num_logits]
        if return_state_hist:
            return logits, state_hist
        else:
            return logits

    def compute_loss(self, x, y, return_state_hist=False, return_logits=True):
        logits, state_hist = self.forward(x, return_state_hist=True)
        if self.num_classes == 2:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        else:
            loss = nn.functional.cross_entropy(logits, y)
        if return_logits and return_state_hist:
            return logits, loss, state_hist
        elif return_logits:
            return logits, loss
        else:
            return loss
        