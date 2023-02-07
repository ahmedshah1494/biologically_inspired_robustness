import time
from typing import Callable, List, Type, Union
import warnings
from attrs import define, field
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
import numpy as np
from positional_encodings.torch_encodings import PositionalEncodingPermute2D

from torch import dropout, nn
import torch
import torchvision

from adversarialML.biologically_inspired_models.src.model_utils import _make_first_dim_last, _make_last_dim_first, merge_strings, mm, str_to_act_and_dact_fn, _compute_conv_output_shape
from adversarialML.biologically_inspired_models.src.supconloss import SupConLoss, AngularSupConLoss
from fastai.vision.models.xresnet import xresnet34, xresnet18, xresnet50, XResNet
from fastai.layers import ResBlock
from einops import rearrange
from fastai.layers import ResBlock
from adversarialML.biologically_inspired_models.src.cornet_s import CORnet_S

from adversarialML.biologically_inspired_models.src.wide_resnet import Wide_ResNet
from adversarialML.biologically_inspired_models.src.FoveatedTextureTransform.model_arch import vgg11_tex_fov
from adversarialML.biologically_inspired_models.src.runners import load_params_into_model
from matplotlib import pyplot as plt

bnrelu = lambda c : nn.Sequential(nn.BatchNorm2d(c),nn.ReLU())
convbnrelu = lambda ci, co, k, s, p: nn.Sequential(nn.Conv2d(ci, co, k, s, p),bnrelu(co))

@define(slots=False)
class CommonModelParams:
    input_size: Union[int, List[int]] = None
    num_units: Union[int, List[int]] = None
    activation: Type[nn.Module] = nn.ReLU
    bias: bool = True
    dropout_p: float = 0.

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
    input_act_consistency_wt:float = 0.3
    no_act_after_update: bool = False

    max_train_time_steps: int = 1
    max_test_time_steps: int = 1

    legacy_act_update_normalization: bool = False
    sparsify_act: bool = False
    sparsity_coeff: float = 1.

    activate_logits: bool = True
    act_opt_mask_p: float = 0.

class CommonModelMixin(object):
    def add_common_params_to_name(self):
        activation_str = self.activation.__str__()
        self.name +=f'-{activation_str[:activation_str.index("(")]}'

        if not self.use_bias:
            self.name += '-noBias'

        if self.sparsify_act:
            self.name += f'-{self.sparsity_coeff:.3f}Sparse'

    def load_common_params(self) -> None:
        input_size = self.params.common_params.input_size
        num_units = self.params.common_params.num_units
        activation = self.params.common_params.activation
        bias = self.params.common_params.bias

        self.input_size = input_size
        self.num_units = num_units
        self.activation = activation
        self.use_bias = bias
        self.dropout_p = self.params.common_params.dropout_p

class ClassificationModelMixin(object):
    def load_classification_params(self) -> None:
        self.num_classes = self.params.classification_params.num_classes
        self.num_logits = (1 if self.num_classes == 2 else self.num_classes)

@define(slots=False)
class ScanningConsistencyOptimizationParams:
    kernel_size: int = 0
    stride: int = 0
    padding: int = 0
    act_opt_kernel_size: int = 0
    act_opt_stride: int = 0
    window_input_act_consistency: bool = True
    spatial_act_act_consistency: bool = False
    use_forward: bool = True


class ConsistencyOptimizationMixin(object):
    def add_consistency_opt_params_to_name(self):
        ld_type = ('Normalized' if self.normalize_lateral_dependence else '')+self.lateral_dependence_type
        _inc_input_str = f'{self.backward_dependence_type}BackDep' if self.input_act_consistency else ''
        self.name = f'Max{ld_type}LatDep{_inc_input_str}{self.name}-{self.num_units}-{self.max_train_time_steps}-{self.max_test_time_steps}steps'

        if self.no_act_after_update:
            self.name += '-noActAfterUpdate'

        if self.truncate_act_opt_grad:
            self.name += '-TruncActOpt'
        if self.use_pt_optimizer:
            self.name += '-PTOptimActOpt'

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
        activate_logits = self.params.consistency_optimization_params.activate_logits
        act_opt_mask_p = self.params.consistency_optimization_params.act_opt_mask_p
        input_act_consistency_wt = self.params.consistency_optimization_params.input_act_consistency_wt

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
        self.activate_logits = activate_logits
        self.act_opt_mask_p = act_opt_mask_p
        self.input_act_consistency_wt = input_act_consistency_wt
        
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
            exit()
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
            
    def _optimization_step(self, step_idx, state, x, W_lat, b_lat, W_back, b_back, activation=None):
        t0 = time.time()
        if (step_idx == 0) or (not self.no_act_after_update):
            if (activation is not None) and isinstance(activation, nn.Module):
                act = activation(state)
            else:
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
            # print(f'{step_idx} max lat update norm = {lat_act_update_norm.max():.3e} max_back_update_norm={back_act_update_norm.max()}')
        act_update_m = lat_act_update + self.input_act_consistency_wt*back_act_update
        loss = lat_loss + back_loss

        with torch.no_grad():
            if self.legacy_act_update_normalization:
                act_update_norm = torch.norm(act_update_m, p=2, dim=0)
            else:
                act_update_norm = act_update_m.reshape(-1,act_update_m.shape[-1]).abs().max(0)[0]
            act_update_norm[act_update_norm == 0] = 1e-8
            scale = self.max_act_update_norm / act_update_norm
            scale = torch.min(scale, torch.ones_like(scale))
            # print(scale.min(), act_update_norm.max(), self.max_act_update_norm)
        act_update_m = act_update_m * scale
        state = act - self.act_opt_step_size * act_update_m
        sparsity_update = self._get_sparsity_update(state)
        state = state - self.act_opt_step_size * sparsity_update
        # print(f"{id(self)}", step_idx, 'loss =', loss.detach().cpu().numpy().mean(), f'max update norm = {act_update_norm.max():.3e} (/{scale.min():.3f})', act_update_m.shape, scale.shape, (state>0).float().mean().cpu().detach().numpy(), time.time() - t0)
        return state, loss, act_update_m

    def _optimize_activations(self, state_hist, x, W_lat, b_lat, W_back, b_back, activation=None):
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
        if not hasattr(self, 'act_opt_mask'):
            self.act_opt_mask = (torch.empty(state_hist[-1].shape[0],1).uniform_() >= self.act_opt_mask_p).float()
            self.act_opt_mask = self.act_opt_mask.to(x.device)
        while (i < max_time_steps) or ((max_time_steps == -1) and (num_bad_steps < max_num_bad_steps)):
            state = state_hist[-1]
            new_state, loss, update = self._optimization_step(i, state, x, W_lat, b_lat, W_back, b_back, activation=activation)            
            if self.act_opt_mask_p > 0:
                new_state = self.act_opt_mask * new_state + (1-self.act_opt_mask)*state
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

class IdentityLayer(AbstractModel):
    def forward(self, x):
        return x

    def compute_loss(self, x, y, return_logits=True):
        logits = x
        loss = torch.zeros((x.shape[0],), device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss    

class ActivationLayer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        activation_cls: Type[nn.Module] = None
    
    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self.activation = self.params.activation_cls()

    def forward(self, x):
        return self.activation(x)

    def compute_loss(self, x, y, return_logits=True):
        logits = x
        loss = torch.zeros((x.shape[0],), device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

class BatchNorm2DLayer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        num_features: int = None
        eps: float = 1e-05
        momentum: float = 0.1
        affine: bool = True
        track_running_stats: bool = True
    
    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self.bn = nn.BatchNorm2d(params.num_features, params.eps, params.momentum, params.affine, params.track_running_stats)
    
    def forward(self, x):
        return self.bn(x)

    def compute_loss(self, x, y, return_logits=True):
        logits = x
        loss = torch.zeros((x.shape[0],), device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss


class ConsistentActivationLayer(AbstractModel, CommonModelMixin, ConsistencyOptimizationMixin):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        consistency_optimization_params: ConsistencyOptimizationParams = field(factory=ConsistencyOptimizationParams)

    def __init__(self, params: ModelParams) -> None:
        print('in ConsistentActivationLayer')
        super().__init__(params)
        self.load_common_params()
        self.input_size = np.prod(self.input_size) if np.iterable(self.input_size) else self.input_size
        self.load_consistency_opt_params()
        self._make_network()
        self._make_name()

    def _make_name(self):
        self.add_common_params_to_name()
        self.add_consistency_opt_params_to_name()

    def _make_network(self):
        state_size = self.num_units + self.input_size
        self.activation = self.activation()
        self.dropout = nn.Dropout(self.dropout_p)
        # self.weight = nn.Parameter(torch.empty((self.num_units, state_size)), True)
        # self.bias = nn.Parameter(torch.empty((self.num_units,)).zero_(), self.use_bias)
        # if self.input_act_consistency:
        #     self.weight_back = nn.Parameter(torch.empty((self.input_size, self.num_units)), True)
        #     self.bias_back = nn.Parameter(torch.empty((self.input_size,)).zero_(), self.use_bias)
        # self.reset_parameters()
        self.fwd = nn.Linear(self.input_size, self.num_units, bias=self.use_bias)
        self.lat = nn.Linear(self.num_units, self.num_units, bias=self.use_bias)
        if self.input_act_consistency:
            self.back = nn.Linear(self.num_units, self.input_size, bias=self.use_bias)

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
        # W = self.weight
        # W_ff = W[:, :self.input_size]
        # W_lat = W[:, self.input_size:]
        # b = self.bias
        # fwd = self._input_to_preact(x.T)
        fwd = self.fwd(x).T
        W_lat = self.lat.weight
        b = self.lat.bias
        b = b.reshape(-1,1)
        state_hist = [fwd]
        # print(0, state_hist[-1][0])
        if self.input_act_consistency:
            # W_back, bias_back = self.weight_back, self.bias_back.reshape(-1, 1)
            W_back = self.back.weight
            bias_back = self.back.bias.reshape(-1, 1)
        else:
            W_back, bias_back = None, None
        self._optimize_activations(state_hist, x.T, W_lat, b, W_back, bias_back)
        logits = state_hist[-1].T
        if self.activate_logits:
            logits = self.activation(logits)
            logits = self.dropout(logits)
        if return_state_hist:
            state_hist = torch.stack(state_hist, dim=1).transpose(0, 2)
            return logits, state_hist
        else:
            return logits

    def compute_loss(self, x, y, return_state_hist=False, return_logits=False):
        logits, state_hist = self.forward(x, return_state_hist=True)
        loss = torch.tensor(0., device=x.device)
        output = (loss,)
        if return_state_hist:
            output = output + (state_hist,)
        if return_logits:
            output = (logits,) + output
        return output

class ConsistentActivationClassifier(ConsistentActivationLayer, ClassificationModelMixin):
    @define(slots=False)
    class ModelParams(ConsistentActivationLayer.ModelParams):
        classification_params: ClassificationModelParams = field(factory=ClassificationModelParams)

    def __init__(self, params: ModelParams) -> None:
        print('in ConsistentActivationClassifier')
        super().__init__(params)
        self.load_classification_params()

    def forward(self, x, return_state_hist=False):
        logits, state_hist = super().forward(x, return_state_hist=True)
        num_logits = (1 if self.num_classes == 2 else self.num_classes)
        logits = logits[:, :num_logits]
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

class ScanningConsistentActivationLayer(AbstractModel, CommonModelMixin, ConsistencyOptimizationMixin):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        consistency_optimization_params: ConsistencyOptimizationParams = field(factory=ConsistencyOptimizationParams)
        scanning_consistency_optimization_params: ScanningConsistencyOptimizationParams = field(factory=ScanningConsistencyOptimizationParams)

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.load_common_params()
        self.load_consistency_opt_params()
        self.load_scanning_consistency_optimization_params()

        self._make_network()
        self._make_name()

    def load_scanning_consistency_optimization_params(self):
        kernel_size: int = self.params.scanning_consistency_optimization_params.kernel_size
        stride: int = self.params.scanning_consistency_optimization_params.stride
        padding: int = self.params.scanning_consistency_optimization_params.padding
        act_opt_kernel_size: int = self.params.scanning_consistency_optimization_params.act_opt_kernel_size
        act_opt_stride: int = self.params.scanning_consistency_optimization_params.act_opt_stride
        window_input_act_consistency: bool = self.params.scanning_consistency_optimization_params.window_input_act_consistency
        spatial_act_act_consistency: bool = self.params.scanning_consistency_optimization_params.spatial_act_act_consistency
        use_forward: bool = self.params.scanning_consistency_optimization_params.use_forward
        if use_forward:
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
        else:
            self.kernel_size = 1
            self.stride = 1
            self.padding = 0
            self.num_units = self.input_size[0]
        self.act_opt_kernel_size = act_opt_kernel_size
        self.act_opt_stride = act_opt_stride
        self.window_input_act_consistency = window_input_act_consistency
        self.spatial_act_act_consistency = spatial_act_act_consistency
        self.use_forward = use_forward

        self.actual_input_size = self.input_size
        self.input_size = self.input_size[0] * (self.kernel_size ** 2)
        self.output_spatial_dims = _compute_conv_output_shape(self.actual_input_size[1:], kernel_size, stride, padding, 1)
    
    def _make_name(self):
        self.add_consistency_opt_params_to_name()
        self.add_common_params_to_name()
        if self.window_input_act_consistency and self.input_act_consistency:
            self.name += '-windowedBack'
        if self.spatial_act_act_consistency:
            self.name = f'Scanning{self.kernel_size}_{self.stride}_{self.padding}-{self.name}'
        else:
            if not self.use_forward:
                self.name = f'Scanning|{self.act_opt_kernel_size}_|{self.act_opt_stride}_-{self.name}'
            else:
                self.name = f'Scanning{self.kernel_size}|{self.act_opt_kernel_size}_{self.stride}|{self.act_opt_stride}_{self.padding}-{self.name}'
    
    def _make_network(self):
        if self.spatial_act_act_consistency:
            output_area = np.prod(self.output_spatial_dims)
            self.lat_layer = nn.Linear(output_area, self.num_units*output_area, bias=self.use_bias)
            self.back_layer = nn.Linear(self.num_units*output_area, np.prod(self.actual_input_size))
        else:
            lat_kernel_volume = self.num_units*(self.act_opt_kernel_size ** 2)
            self.lat_layer = nn.Linear(lat_kernel_volume, lat_kernel_volume, bias=self.use_bias)
            if self.input_act_consistency:
                if self.window_input_act_consistency:
                    inp_kernel_volume =  self.actual_input_size[0]*((self.kernel_size + (self.act_opt_kernel_size-1)*self.stride)**2)
                    self.back_layer = nn.Linear(lat_kernel_volume, inp_kernel_volume)
                else:
                    inp_kernel_volume = self.actual_input_size[0]*(self.kernel_size ** 2)
                    self.back_layer = nn.Linear(self.num_units, inp_kernel_volume)
        if self.use_forward:
            self.fwd_layer = nn.Conv2d(self.actual_input_size[0], self.num_units, self.kernel_size, self.stride, 
                                            self.padding, bias=self.use_bias)
        else:
            self.fwd_layer = nn.Identity()
        self.activation = self.params.common_params.activation()
        self.dropout = nn.Dropout(self.dropout_p)
        self._maybe_create_overlap_count_map()
    
    def _get_num_padding(self, n, k, s):
        p = ((k - ((n - k) % s)) % k) // 2
        return p

    def _maybe_create_overlap_count_map(self):
        x = torch.rand(1,*self.actual_input_size)
        fwd = self.fwd_layer(x)
        bs, c, h, w = fwd.shape
        if self.act_opt_kernel_size > self.act_opt_stride:
            overlap = nn.functional.fold(
                nn.functional.unfold(
                                        torch.ones(fwd.shape), self.act_opt_kernel_size, 
                                        stride=self.act_opt_stride, 
                                        padding=self._get_num_padding(h, self.act_opt_kernel_size, self.act_opt_stride)
                                    ),
                (h, w), self.act_opt_kernel_size, stride=self.act_opt_stride, padding=self._get_num_padding(h, self.act_opt_kernel_size, self.act_opt_stride)
            )
            self.overlap = nn.Parameter(overlap, False)
        else:
            self.overlap = None

    def _unfold_for_act_opt(self, y):
        c, h, w, bs = y.shape
        y = _make_last_dim_first(y)        
        y_unf = nn.functional.unfold(y, self.act_opt_kernel_size, stride=self.act_opt_stride,
                                        padding=self._get_num_padding(h, self.act_opt_kernel_size, self.act_opt_stride))
        return y_unf

    def _fold_for_act_opt(self, y, y_orig_shape):
        c, h, w, bs = y_orig_shape
        block_dim, bs_x_nblocks = y.shape
        y_unf = y.reshape(block_dim, bs, -1)
        y_unf = y_unf.transpose(1,0)
        y_fld = nn.functional.fold(y_unf, (h, w), self.act_opt_kernel_size, stride=self.act_opt_stride,
                                    padding=self._get_num_padding(h, self.act_opt_kernel_size, self.act_opt_stride))
        return y_fld

    def _reshape_input_for_act_act_consistency_optimization(self, y):
        if self.spatial_act_act_consistency:
            return y

        if y.dim() == 4:
            y_unf = self._unfold_for_act_opt(y)
            bs, block_dim, num_blocks = y_unf.shape
            y_unf = y_unf.transpose(0,1)
            y_unf = y_unf.reshape(block_dim, -1)
        else:
            block_dim, num_blocks, bs = y.shape
            y = y.transpose(1,2)
            y_unf = y.reshape(block_dim, -1)
        return y_unf

    def _reshape_output_for_act_act_consistency_optimization(self, y, y_orig_shape):
        if self.spatial_act_act_consistency:
            return y
        if (not self.input_act_consistency) or self.window_input_act_consistency:
            block_dim, num_blocks, bs = y_orig_shape
            y = y.reshape(block_dim, bs, num_blocks)
            y = y.transpose(1,2)
            return y
        y_fld = self._fold_for_act_opt(y, y_orig_shape)
        if self.overlap is not None:
            y_fld = y_fld / self.overlap
        y_fld = _make_first_dim_last(y_fld)
        return y_fld

    def _get_activation_update_manual(self, act: torch.Tensor, W: torch.Tensor, b: torch.Tensor, normalize_by_act_norm=False):
        act_shape = act.shape
        act = self._reshape_input_for_act_act_consistency_optimization(act)
        act_update, loss = super()._get_activation_update_manual(act, W, b, normalize_by_act_norm)
        act_update = self._reshape_output_for_act_act_consistency_optimization(act_update, act_shape)
        return act_update, loss

    def _reshape_input_for_input_act_consistency_optimization(self, y):
        if self.spatial_act_act_consistency:
            c, h_w, bs = y.shape
            y = y.reshape(-1, bs)
        else:
            if self.window_input_act_consistency:
                block_dim, num_blocks, bs = y.shape
                y = y.reshape(block_dim, -1)
            else:
                c, h, w, bs = y.shape
                y = y.reshape(c, -1)
        return y
    
    def _reshape_output_for_input_act_consistency_optimization(self, y, y_orig_shape):
        if self.spatial_act_act_consistency or self.window_input_act_consistency:
            y = y.reshape(*y_orig_shape)
        else:
            c, h, w, bs = y_orig_shape
            c, h_w_bs = y.shape
            y = y.reshape(c, h, w, bs)
        return y

    def _get_activation_backward_update_manual(self, x: torch.Tensor, act: torch.Tensor, W: torch.Tensor, b: torch.Tensor, normalize_by_act_norm=False):
        act_shape = act.shape
        act = self._reshape_input_for_input_act_consistency_optimization(act)
        act_update, loss = super()._get_activation_backward_update_manual(x, act, W, b, normalize_by_act_norm)
        act_update = self._reshape_output_for_input_act_consistency_optimization(act_update, act_shape)
        return act_update, loss

    def prepare_inputs_for_optimization(self, state_hist, x):
        fwd = state_hist[-1]
        bs, c, h, w = fwd.shape
        if self.spatial_act_act_consistency:
            fwd = fwd.reshape(fwd.shape[0], fwd.shape[1], -1)
            fwd = fwd.transpose(0,1).transpose(1,2)
            x = x.reshape(x.shape[0], -1).T
            W_lat = W_lat.reshape(self.num_units, -1, W_lat.shape[1])
            b_lat = b_lat.reshape(self.num_units, -1, 1)
            state_hist[-1] = fwd
        else:
            state_hist[-1] = _make_first_dim_last(fwd)
            if self.input_act_consistency:
                x_unf = nn.functional.unfold(x, self.kernel_size, stride=self.stride, padding=self.padding)
                if self.window_input_act_consistency:
                    k = self.kernel_size + (self.act_opt_kernel_size-1)*self.stride
                    s = self.stride * self.act_opt_stride
                    h = int(np.floor((x.shape[1] + 2*self.padding - (self.kernel_size-1) - 1)/self.stride + 1))
                    p = self._get_num_padding(h, self.act_opt_kernel_size, self.act_opt_stride)
                    x_unf = nn.functional.unfold(x, k, stride=s, padding=self.padding+p)
                    state_hist[-1] = _make_first_dim_last(self._unfold_for_act_opt(state_hist[-1]))
                bs, block_dim, num_blocks = x_unf.shape
                # x_unf = x_unf.transpose(0,1)
                x_unf = _make_first_dim_last(x_unf)
                x = x_unf.reshape(block_dim, -1)
            else:
                state_hist[-1] = _make_first_dim_last(self._unfold_for_act_opt(state_hist[-1]))
        return state_hist, x

    def _optimize_activations(self, state_hist, x, W_lat, b_lat, W_back, b_back):
        state_hist, x = self.prepare_inputs_for_optimization(state_hist, x)
        super()._optimize_activations(state_hist, x, W_lat, b_lat, W_back, b_back)

    def reshape_state(self, s, orig_shape):
        if self.input_act_consistency and (not self.window_input_act_consistency):
            s = _make_last_dim_first(s)
        else:
            s = s.transpose(1,2).reshape(s.shape[0], -1)
            s = self._fold_for_act_opt(s, orig_shape)
            s = s.reshape(s.shape[0], s.shape[1], *(self.output_spatial_dims))
        return s

    def forward(self, x, return_state_hist=False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        fwd = self.fwd_layer(x)
        W_lat = self.lat_layer.weight
        b_lat = self.lat_layer.bias.reshape(-1,1)
        state_hist = [fwd]
        W_back = self.back_layer.weight if self.input_act_consistency else None
        b_back = self.back_layer.bias.reshape(-1, 1) if self.input_act_consistency else None
        self._optimize_activations(state_hist, x, W_lat, b_lat, W_back, b_back)
        orig_shape = _make_first_dim_last(fwd).shape
        logits = self.reshape_state(state_hist[-1], orig_shape)
        if self.activate_logits:
            logits = self.activation(logits)
            logits = self.dropout(logits)
        if return_state_hist:
            state_hist = [self.reshape_state(s, orig_shape) for s in state_hist]
            state_hist = torch.stack(state_hist, dim=1)
            return logits, state_hist
        else:
            return logits
    
    def compute_loss(self, x, y, return_state_hist=False, return_logits=False):
        logits, state_hist = self.forward(x, return_state_hist=True)
        loss = torch.tensor(0., device=x.device)
        output = (loss,)
        if return_state_hist:
            output = output + (state_hist,)
        if return_logits:
            output = (logits,) + output
        return output

@define(slots=False)
class PositionalEmbeddingParams:
    pos_emb_cls: Type = None
    cat_emb: bool = False

class LearnablePositionEmbedding(nn.Module):
    def __init__(self, input_shape) -> None:
        super().__init__()
        emb = torch.empty(input_shape).unsqueeze(0).uniform_(-0.05, 0.05)
        self.emb = nn.parameter.Parameter(emb, True)
    
    def forward(self, x):
        return self.emb

class PositionAwareScanningConsistentActivationLayer(ScanningConsistentActivationLayer):
    @define(slots=False)
    class ModelParams(ScanningConsistentActivationLayer.ModelParams):
        position_embedding_params: PositionalEmbeddingParams = field(factory=PositionalEmbeddingParams)
    
    def _make_network(self):
        if issubclass(self.params.position_embedding_params.pos_emb_cls, PositionalEncodingPermute2D):
            self.pos_emb = self.params.position_embedding_params.pos_emb_cls(self.actual_input_size[0])
        elif issubclass(self.params.position_embedding_params.pos_emb_cls, LearnablePositionEmbedding):
            self.pos_emb = self.params.position_embedding_params.pos_emb_cls(self.actual_input_size)
        if self.params.position_embedding_params.cat_emb:
            self.actual_input_size = list(self.actual_input_size)
            self.actual_input_size[0] *= 2
            self.input_size = self.actual_input_size[0] * (self.kernel_size ** 2)
        return super()._make_network()
    
    def forward(self, x, return_state_hist=False):
        pe = self.pos_emb(x)
        if issubclass(self.params.position_embedding_params.pos_emb_cls, PositionalEncodingPermute2D):
            pe *= 0.05
        if self.params.position_embedding_params.cat_emb:
            if pe.shape[0] == 1:
                pe = pe.repeat(x.shape[0], 1, 1, 1)
            x = torch.cat((x, pe), dim=1)
        else:
            x = x + pe
        return super().forward(x, return_state_hist)
    
class SequentialLayers(AbstractModel, CommonModelMixin):
    @define(slots=False)
    class ModelParams(BaseParameters):
        layer_params: List[BaseParameters] = field(factory=list)
        common_params: CommonModelParams = field(factory=CommonModelParams)
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self.load_common_params()
        self._make_network()
        self._make_name()
    
    def _make_name(self):
        self.name = merge_strings([l.name for l in self.layers])
        self.name += f'-{self.params.common_params.dropout_p}Dropout'
    
    def _make_network(self):
        if self.params.common_params.input_size is not None:
            input_size = self.params.common_params.input_size
        else:
            input_size = self.params.layer_params[0].common_params.input_size
        x = torch.rand(1,*input_size)
        layers = []
        print(x.shape)
        for lp in self.params.layer_params:
            if hasattr(lp, 'common_params'):
                lp.common_params.input_size = x.shape[1:]
            l = lp.cls(lp)
            x = l(x)
            if isinstance(x, tuple):
                x = x[0]
            print(type(l), x.shape)
            layers.append(l)
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, *fwd_args, **fwd_kwargs):
        out = x
        extra_outputs = []
        for l in self.layers:
            if self.training:
                out = l.compute_loss(out, None, return_logits=True)
            else:
                out = l(out, *fwd_args, **fwd_kwargs)
            if isinstance(out, tuple):
                extra_outputs.append(out[1:])
                out = out[0]
        if len(extra_outputs) > 0:
            return out, extra_outputs
        else:
            return out
    
    def compute_loss(self, x, y, return_logits=True, **fwd_kwargs):
        out = self.forward(x, **fwd_kwargs)
        if isinstance(out, tuple):
            logits = out[0]
            loss = out[1][0]
        else:
            logits = out
            loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

class ConcurrentConsistencyOptimizationSequentialLayers(SequentialLayers):
    def _make_network(self):
        super()._make_network()

        act_opt_layers = [l for l in self.layers if isinstance(l, ConsistencyOptimizationMixin)]
        if len(act_opt_layers) > 0:
            self.max_train_time_steps = max([l.max_train_time_steps for l in act_opt_layers])
            self.max_test_time_steps = max([l.max_test_time_steps for l in act_opt_layers])
        for l in act_opt_layers:
            l.max_train_time_steps = l.max_test_time_steps = 0
        
        self.states = [None]*len(self.layers)

    def forward_step(self, step_idx, x, current_updates, *fwd_args, **fwd_kwargs):
        out = x
        extra_outputs = []
        new_updates = []
        states = [None]*len(self.layers)
        for layer_idx, (l, u, prev_state) in enumerate(zip(self.layers, current_updates, states)):
            update = 0
            if self.training:
                if isinstance(l, ScanningConsistentActivationLayer):
                    _, loss, state_hist = l.compute_loss(out, None, return_logits=True, return_state_hist=True)
                    state = state_hist[:,0]
                    
                    if prev_state is not None:
                        W_lat = l.lat_layer.weight
                        b_lat = l.lat_layer.bias.reshape(-1,1)
                        W_back = l.back_layer.weight if l.input_act_consistency else None
                        b_back = l.back_layer.bias.reshape(-1, 1) if l.input_act_consistency else None
                        [prev_state], x = l.prepare_inputs_for_optimization([prev_state], out)
                        new_prev_state, loss, update = l._optimization_step(step_idx, prev_state, x, W_lat, b_lat, W_back, b_back)
                    
                        orig_shape = _make_first_dim_last(state).shape
                        prev_state = l.reshape_state(prev_state, orig_shape)
                        new_prev_state = l.reshape_state(new_prev_state, orig_shape)
                        update = (new_prev_state - prev_state)
                        update_norm = update.reshape(-1,update.shape[-1]).abs().max(0)[0]

                        state = state + update
                    states[layer_idx] = state
                    logits = l.dropout(l.activation(state))
                    out = (logits, loss)
                else:
                    out = l.compute_loss(out, None, return_logits=True)
            else:
                out = l(out, *fwd_args, **fwd_kwargs)
            new_updates.append(update)
            if isinstance(out, tuple):
                extra_outputs.append(out[1:])
                out = out[0]
        return out, new_updates, extra_outputs
    
    def forward(self, x, *fwd_args, **fwd_kwargs):
        max_num_steps = self.max_train_time_steps if self.training else self.max_test_time_steps
        out = x
        updates = [0]*len(self.layers)
        for i in range(max_num_steps):
            out, updates, extra_outputs = self.forward_step(i, x, updates, *fwd_args, **fwd_kwargs)
        if len(extra_outputs) > 0:
            return out, extra_outputs
        else:
            return out

class GeneralClassifier(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        input_size: List[int] = None
        feature_model_params: BaseParameters = None
        classifier_params: BaseParameters = None
        logit_ensembler_params: BaseParameters = None
        loss_fn: nn.Module = nn.CrossEntropyLoss
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self._make_network()
        self._make_name()
    
    def _make_name(self):
        self.name = self.feature_model.name
    
    def _make_network(self):
        fe_params = self.params.feature_model_params
        self.feature_model = fe_params.cls(fe_params)
        input_size = self.params.input_size if self.params.input_size is not None else fe_params.common_params.input_size
        x = torch.rand(1,*(input_size))
        x = self.feature_model(x)
        if isinstance(x, tuple):
            x = x[0]

        cls_params = self.params.classifier_params
        if hasattr(cls_params, 'common_params'):
            cls_params.common_params.input_size = x.shape[1:]
        else:
            cls_params.input_size = x.shape[1:]
        self.classifier = cls_params.cls(cls_params)
        if self.params.logit_ensembler_params is not None:
            self.logit_ensembler = self.params.logit_ensembler_params.cls(self.params.logit_ensembler_params)
        else:
            self.logit_ensembler = nn.Identity()
        self.loss_fn = self.params.loss_fn()

    def _get_feats(self, x):
        r = self.feature_model.forward(x)
        if isinstance(r, tuple):
            r = r[0]
        return r
    
    def _run_classifier(self, x):
        logits = self.classifier(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = self.logit_ensembler(logits)
        return logits
    
    def forward(self, x, *fwd_args, **fwd_kwargs):
        y = self.feature_model.forward(x, *fwd_args, **fwd_kwargs)
        extra_outputs = []
        if isinstance(y, tuple):
            r = y[0]
            conv_extra_outputs = y[1]
            extra_outputs.extend(conv_extra_outputs)
        else:
            r = y
            conv_extra_outputs = ()
        # if np.random.uniform(0, 1) < 1/100:
        #     _r = r.detach().cpu().numpy()
        #     print(f'sparsity={(_r == 0).astype(float).mean():3f}', _r.max(), np.quantile(_r, 0.99))
        out = self.classifier(r, *fwd_args, **fwd_kwargs)
        if isinstance(out, tuple):
            logits = out[0]
            cls_extra_outputs = out[1:]
            extra_outputs.append(cls_extra_outputs)
        else:
            logits = out
            cls_extra_outputs = ()
        logits = self.logit_ensembler(logits)
        if len(cls_extra_outputs) == 0 and len(conv_extra_outputs) == 0:
            return logits
        extra_outputs = list(zip(*extra_outputs))
        return (logits, *extra_outputs)

    def compute_loss(self, x, y, *fwd_args, return_logits=True, **fwd_kwargs):
        out = self.forward(x, *fwd_args, **fwd_kwargs)
        if isinstance(out, tuple):
            logits = out[0]
            extra_outputs = out[1:]
        else:
            logits = out
            extra_outputs = ()
        # if (logits.dim() == 1) or ((logits.dim() > 1) and (logits.shape[-1] == 1)):
        #     loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        # else:
        #     loss = nn.functional.cross_entropy(logits, y)
        loss = self.loss_fn(logits, y)
        if len(extra_outputs) > 0:
            loss = loss #+ torch.stack(extra_outputs[0], 0).sum(0)
            extra_outputs = extra_outputs[1:]
            assert loss.dim() == 0
        outputs = []
        if return_logits:
            outputs.append(logits)
        outputs.append(loss)
        if len(extra_outputs) > 0:
            outputs.extend(extra_outputs)
        return tuple(outputs)

class EyeModel(AbstractModel, CommonModelMixin):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = None
        photoreceptor_params: BaseParameters = None
        horizontal_cell_params: BaseParameters = None
        bipolar_cell_params: BaseParameters = None
        amacrine_cell_params: BaseParameters = None
        ganglion_cell_params: BaseParameters = None
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.load_common_params()
        self._make_network()
        self._make_name()
    
    def _make_network(self):
        x = torch.rand((1, *(self.input_size)))
        self.photoreceptors = self.params.photoreceptor_params.cls(self.params.photoreceptor_params)
        x = self.photoreceptors.eval()(x)
        self.params.horizontal_cell_params.common_params.input_size = x.shape[1:]
        self.h_cells = self.params.horizontal_cell_params.cls(self.params.horizontal_cell_params)
        x = self.h_cells.eval()(x)
        self.params.bipolar_cell_params.common_params.input_size = x.shape[1:]
        self.bp_cells = self.params.bipolar_cell_params.cls(self.params.bipolar_cell_params)
        x= self.bp_cells.eval()(x)
        self.params.amacrine_cell_params.common_params.input_size = x.shape[1:]
        self.params.ganglion_cell_params.common_params.input_size = x.shape[1:]
        self.a_cells = self.params.amacrine_cell_params.cls(self.params.amacrine_cell_params)
        self.g_cells = self.params.ganglion_cell_params.cls(self.params.ganglion_cell_params)

        self.dropout = nn.Dropout2d(self.params.common_params.dropout_p)
        self.activation = self.activation()

    def forward(self, x, *fwd_args, **fwd_kwargs):
        out = x
        extra_outputs = []

        def forward_and_maybe_loss(l, x):
            if self.training:
                out = l.compute_loss(x, None, return_logits=True)
            else:
                out = l(x, *fwd_args, **fwd_kwargs)
            if isinstance(out, tuple):
                extra_outputs.append(out[1:])
                out = out[0]
            return out

        def maybe_acitvate(l, x):
            if not isinstance(l, SequentialLayers):
                x = self.activation(x)
            return x

        def maybe_dropout(l, x):
            if not isinstance(l, SequentialLayers):
                x = self.dropout(x)
            return x

        out = forward_and_maybe_loss(self.photoreceptors, out)
        # out = maybe_acitvate(self.photoreceptors, out)

        out = forward_and_maybe_loss(self.h_cells, out)
        # out = maybe_acitvate(self.h_cells, out)

        out = forward_and_maybe_loss(self.bp_cells, out)
        # out = maybe_acitvate(self.bp_cells, out)
        # out = maybe_dropout(self.bp_cells, out)
        
        a_proj = forward_and_maybe_loss(self.a_cells, out)
        # a_proj = maybe_acitvate(self.a_cells, a_proj)
        # a_proj = maybe_dropout(self.a_cells, a_proj)
        
        out = forward_and_maybe_loss(self.g_cells, a_proj + out)
        # out = maybe_acitvate(self.g_cells, out)
        # out = maybe_dropout(self.g_cells, out)
        
        if len(extra_outputs) > 0:
            return out, extra_outputs
        else:
            return out
    
    def compute_loss(self, x, y, return_logits=True):
        out = self.forward(x)
        if isinstance(out, tuple):
            logits = out[0]
            loss = out[1][0]
        else:
            logits = out
            loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss
        

@define(slots=False)
class ConvParams:
    kernel_sizes: List[int] = None
    strides: List[int] = None
    padding: List[int] = None

class ConvEncoder(AbstractModel, CommonModelMixin):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        conv_params: ConvParams = field(factory=ConvParams)

    def __init__(self, params: ModelParams) -> None:
        super(ConvEncoder, self).__init__(params)
        self.params = params
        self.load_common_params()
        self._load_conv_params()
        self._make_name()
        self._make_network()

    def _load_conv_params(self):
        self.kernel_sizes = self.params.conv_params.kernel_sizes
        self.strides = self.params.conv_params.strides
        self.padding = self.params.conv_params.padding
    
    def _make_name(self):
        layer_desc = [f'{f}x{ks}_{s}_{p}' for ks,s,p,f in zip(self.kernel_sizes, self.strides, self.padding, self.num_units)]
        layer_desc_2 = []
        curr_desc = ''
        for i,x in enumerate(layer_desc):
            if x == curr_desc:
                count += 1
            else:
                if curr_desc != '':
                    layer_desc_2.append(f'{count}x{curr_desc}')
                count = 1
                curr_desc = x
            if i == len(layer_desc)-1:
                layer_desc_2.append(f'{count}x{curr_desc}')
        self.name = 'Conv-'+'_'.join(layer_desc_2)
    
    def _make_network(self):
        layers = []
        nfilters = [self.input_size[0], *self.num_units]
        kernel_sizes = [None] + self.kernel_sizes
        strides = [None] + self.strides
        padding = [None] + self.padding
        for i, (k,s,f,p) in enumerate(zip(kernel_sizes, strides, nfilters, padding)):
            if i > 0:
                layers.append(nn.Conv2d(nfilters[i-1], f, k, s, p, bias=self.use_bias))
                layers.append(self.activation())
                if self.dropout_p > 0:
                    layers.append(nn.Dropout2d(self.dropout_p))
        self.conv_model = nn.Sequential(*layers)
    
    def forward(self, x, return_state_hist=False, **kwargs):
        feat = self.conv_model(x)
        return feat
    
    def compute_loss(self, x, y, return_state_hist=False, return_logits=False):
        logits = self.forward(x, return_state_hist=True)
        loss = torch.tensor(0., device=x.device)
        output = (loss,)
        if return_state_hist:
            output = output + (None,)
        if return_logits:
            output = (logits,) + output
        return output

class SupervisedContrastiveTrainingWrapper(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        model_params: BaseParameters = None
        projection_params: BaseParameters = None
        use_angular_supcon_loss: bool = False
        angular_supcon_loss_margin: float = 1.
        supcon_loss_temperature: float = 0.07
    
    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self._make_network()

    def _make_network(self):
        super()._make_network()
        self.base_model = self.params.model_params.cls(self.params.model_params)
        self.proj = self.params.projection_params.cls(self.params.projection_params)
        if self.params.use_angular_supcon_loss:
            self.lossfn = AngularSupConLoss(base_temperature=self.params.supcon_loss_temperature, temperature=self.params.supcon_loss_temperature, margin=self.params.angular_supcon_loss_margin)
        else:
            self.lossfn = SupConLoss(base_temperature=self.params.supcon_loss_temperature, temperature=self.params.supcon_loss_temperature)
    
    def _get_proj(self, z):
        if not isinstance(self.proj, IdentityLayer):
            p = self.proj(z)
            p = nn.functional.normalize(p)
        else:
            p = z
        return p
    
    def _get_feats(self, x):
        z = self.base_model._get_feats(x)
        z = nn.functional.normalize(z)
        return z
    
    def _run_classifier(self, x):
        return self.base_model._run_classifier(x)
    
    def forward(self, x):
        return self._run_classifier(self._get_feats(x)) 
    
    def compute_loss(self, x, y, return_logits=True):
        if x.dim() == 5:
            x = rearrange(x, 'b n c h w -> (n b) c h w')
        feat = self._get_feats(x)
        proj = self._get_proj(feat)
        logits = self._run_classifier(feat.detach())

        proj = rearrange(proj, '(n b) d -> b n d', b=y.shape[0])
        logits = rearrange(logits, '(n b) d -> b n d', b=y.shape[0])

        loss1 = self.lossfn(proj, y)
        
        loss2 = 0
        for i in range(logits.shape[1]):
            loss2 += nn.functional.cross_entropy(logits[:, i], y)
        loss2 /= logits.shape[1]
        loss = loss1 + loss2

        logits = logits.mean(1)
        if return_logits:
            return logits, loss
        else:
            return loss

class XResNet34(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        normalization_layer_params: BaseParameters = None
        setup_feature_extraction: bool = False
        setup_classification: bool = True
        num_classes: int = None
        kernel_size: int = 3
        widen_factor: int = 1.
        widen_stem: bool = False

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params: XResNet34.ModelParams = params
        self._make_network()

    def _make_resnet(self):
        resnet = xresnet34(p=self.params.common_params.dropout_p,
                                c_in=self.params.common_params.input_size[0],
                                n_out=self.params.common_params.num_units,
                                act_cls=self.params.common_params.activation,
                                widen=self.params.widen_factor,
                                stem_szs=(32,32,64) if not self.params.widen_stem else (self.params.widen_factor*32,self.params.widen_factor*32,64),
                                ks=self.params.kernel_size
                            )
        resnet[-1] = nn.Identity()
        return resnet

    def _make_network(self):
        if self.params.normalization_layer_params is not None:
            self.normalization_layer = self.params.normalization_layer_params.cls(self.params.normalization_layer_params)
        else:
            self.normalization_layer = nn.Identity()
        self.resnet = self._make_resnet()
        if self.params.setup_classification:
            x = self.resnet(torch.rand(1, *(self.params.common_params.input_size)))
            self.classifier = nn.Linear(x.shape[1], self.params.num_classes)

    def _get_feats(self, x, **kwargs):
        x = self.normalization_layer(x)
        feat = self.resnet(x)
        return feat
    
    def _run_classifier(self, x):
        return self.classifier(x)
    
    def forward(self, x, **kwargs):
        x = self._get_feats(x)
        if self.params.setup_classification:
            x =  self._run_classifier(x)
        return x
    
    def compute_loss(self, x, y, return_logits=True, **kwargs):
        logits = self.forward(x)
        if self.params.setup_classification:
            loss = nn.functional.cross_entropy(logits, y)
        else:
            loss = torch.tensor(0., device=x.device)
        # if self.training:
        #     print(logits[:2], loss)
        output = (loss,)
        if return_logits:
            output = (logits,) + output
        return output

class XResNet18(XResNet34):
    def _make_resnet(self):
        resnet = xresnet18(p=self.params.common_params.dropout_p,
                                c_in=self.params.common_params.input_size[0],
                                n_out=self.params.common_params.num_units,
                                act_cls=self.params.common_params.activation,
                                widen=self.params.widen_factor,
                                stem_szs=(32,32,64) if not self.params.widen_stem else (self.params.widen_factor*32,self.params.widen_factor*32,64),
                                ks=self.params.kernel_size
                            )
        resnet[-1] = nn.Identity()
        return resnet

class XResNet50(XResNet34):
    def _make_resnet(self):
        resnet = xresnet50(p=self.params.common_params.dropout_p,
                                c_in=self.params.common_params.input_size[0],
                                n_out=self.params.common_params.num_units,
                                act_cls=self.params.common_params.activation,
                                widen=self.params.widen_factor,
                                stem_szs=(32,32,64) if not self.params.widen_stem else (self.params.widen_factor*32,self.params.widen_factor*32,64),
                                ks=self.params.kernel_size
                            )
        resnet[-1] = nn.Identity()
        return resnet

class CORnetS(XResNet34):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        normalization_layer_params: BaseParameters = None
        setup_feature_extraction: bool = False
        setup_classification: bool = True
        num_classes: int = None
        num_recurrence: List[int] = [2,4,2]

    def _make_resnet(self):
        resnet = CORnet_S(times=self.params.num_recurrence)
        return resnet

class WideResnet(XResNet34):
    @define(slots=False)
    class ModelParams(XResNet34.ModelParams):
        depth: int = None
        widen_factor: int = None
    
    def _make_resnet(self):
        resnet = Wide_ResNet(self.params.depth, self.params.widen_factor, self.params.common_params.dropout_p, self.params.num_classes)
        resnet.linear = nn.Identity()
        return resnet

class LogitAverageEnsembler(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        n: int = None
        activation: nn.Module = nn.Identity
        reduction: str = 'mean'
    
    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self.activation = self.params.activation()

    def __repr__(self):
        return f'LogitAverageEnsembler(n={self.params.n}, act={self.activation})'

    def forward(self, x):
        # Assumes that all the logits from the n instances are consecuitve i.e.
        # x = x_.reshape(-1, C) where x_.shape = [b, n, C]
        bn, c = x.shape
        x = self.activation(x)
        if (bn < self.params.n) or (bn % self.params.n != 0):
            w = f'Expected the size of x at dim 0 to be a non-zero multiple of {self.params.n} but got {bn}. Returning x as is.'
            warnings.warn(w)
        else:
            x = rearrange(x, '(b n) c -> b n c', n=self.params.n)
        if self.params.reduction == 'mean':
            x = x.mean(1)
        elif self.params.reduction == 'logsumexp':
            x = torch.logsumexp(x, 1)
        return x
    
    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        output = (loss,)
        if return_logits:
            output = (logits,) + output
        return output

class FovTexVGG(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        scale: str = '0.4'
        num_classes: int = None
        permutation: str = None

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()

    def _make_network(self):
        self.vgg = vgg11_tex_fov(self.params.scale, self.params.common_params.input_size[1], 
                                    self.params.num_classes, self.params.permutation)

    def forward(self, x):
        return self.vgg(x)
    
    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        output = (loss,)
        if return_logits:
            output = (logits,) + output
        return output

class XResNetClassifierWithReconstructionLoss(XResNet18):
    @define(slots=False)
    class ModelParams(XResNet18.ModelParams):
        preprocessing_params: BaseParameters = None
        recon_wt: float = 1.
        cls_wt: float = 1.
        feature_layer_idx: int = -1
        logit_ensembler_params: BaseParameters = None

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        params.setup_classification = params.setup_feature_extraction = True
        self._make_network()

    def _make_reconstructor(self):
        isz = self.params.common_params.input_size
        shapes = [(1, *isz)]
        isz = shapes[-1]
        for i, block in enumerate(self.resnet):
            if i <= self.params.feature_layer_idx:
                x = torch.rand(*isz)
                x = block(x)
                isz = x.shape
                shapes.append(isz)
        recon_layers = []
        self.combiners = nn.ModuleList([])
        for i in range(len(shapes)-1, 0, -1):
            s = shapes[i-1][2]//shapes[i][2]
            l = nn.ConvTranspose2d(shapes[i][1], shapes[i-1][1], 3, s, 1, int(s > 1))
            # l = nn.Conv2d(shapes[i][1], shapes[i-1][1]*(s**2), 3, 1, 1)
            # if s > 1:
            #     l = nn.Sequential(l, nn.PixelShuffle(s))
            x = l(x)
            recon_layers.append(l)
            if i < len(shapes)-1:
                c = nn.Conv2d(2*shapes[i][1], shapes[i][1], 1, 1, 0)
                self.combiners.append(c)
        self.reconstructor = nn.Sequential(*recon_layers)
    
    def _make_network(self):
        super()._make_network()
        if self.params.preprocessing_params is None:
            self.preprocessor = nn.Identity()
        else:
            self.preprocessor = self.params.preprocessing_params.cls(self.params.preprocessing_params)
        self._make_reconstructor()
        if self.params.logit_ensembler_params is not None:
            self.logit_ensembler = self.params.logit_ensembler_params.cls(self.params.logit_ensembler_params)
        else:
            self.logit_ensembler = nn.Identity()
    
    def preprocess(self, x):
        x = self.preprocessor(x)
        if isinstance(x, tuple):
            x = x[0]
        return x
    
    def _get_feats_(self, x, return_interm_activations=False, **kwargs):
        x = self.normalization_layer(x)
        feats = []
        for i,l in enumerate(self.resnet):
            if i <= self.params.feature_layer_idx:
                x = l(x)
                feats.append(x)
        if return_interm_activations:
            return x, feats
        return x
    
    def _get_feats(self, x, return_interm_activations=False, **kwargs):
        x = self.preprocess(x)
        return self._get_feats_(x, return_interm_activations=return_interm_activations, **kwargs)
    
    def _run_classifier(self, x):
        for i,l in enumerate(self.resnet):
            if i > self.params.feature_layer_idx:
                x = l(x)
        logits = self.classifier(x)
        logits = self.logit_ensembler(logits)
        return logits

    def forward_and_reconstruct(self, x):
        x = self.preprocess(x)
        feats, all_feats = self._get_feats_(x, return_interm_activations=True)
        if self.params.recon_wt > 0:
            all_feats = all_feats[::-1]
            for i,l in enumerate(self.reconstructor):
                f = all_feats[i]
                if i > 0:
                    f = torch.relu(self.combiners[i-1](torch.cat([r, f], 1)))
                r = l(f)
                if i < (len(self.reconstructor)-1):
                    r = torch.relu(r)
        else:
            r = torch.zeros_like(x)
        if self.params.cls_wt > 0:
            logits = self._run_classifier(feats)
        else:
            logits = torch.zeros((x.shape[0], self.params.num_classes), dtype=x.dtype, device=x.device)
        plt.subplot(1,2,1)
        plt.imshow(convert_image_tensor_to_ndarray(x[0]))
        plt.subplot(1,2,2)
        plt.imshow(convert_image_tensor_to_ndarray(r[0]))
        plt.savefig('recon2.png')
        return logits, r
    
    def compute_loss(self, x, y, return_logits=True):
        if (self.params.recon_wt > 0):# and self.training:
            logits, recon = self.forward_and_reconstruct(x)
            rloss = torch.pow(x.reshape(x.shape[0],-1) - recon.reshape(recon.shape[0],-1), 2).mean()
        else:
            logits = self.forward(x)
            rloss = 0
        closs = nn.functional.cross_entropy(logits, y)
        loss = self.params.cls_wt * closs + self.params.recon_wt * rloss
        if return_logits:
            return logits, loss
        return loss

def convert_image_tensor_to_ndarray(img):
    return img.cpu().detach().transpose(0,1).transpose(1,2).numpy()

class XResNetClassifierWithEnhancer(XResNetClassifierWithReconstructionLoss):
    @define(slots=False)
    class ModelParams(XResNetClassifierWithReconstructionLoss.ModelParams):
        no_reconstruction: bool = False
        use_residual_during_inference: bool = False

    def _make_reconstructor(self):
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(3, 32, 9, 4, 4),
            nn.Conv2d(3, 32, 7, 4, 3),
            nn.Conv2d(3, 32, 5, 4, 2),
        ])
        self.bnrelu = nn.Sequential(
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.reconstructor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(96, 64, 5, 1, 2),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(160, 64, 1, 1),
                nn.ReLU(),
                nn.Conv2d(64, 3*16, 5, 1, 2),
                nn.PixelShuffle(4)
            ),
        ])

    def reconstruct(self, x):
        z1 = self.bnrelu(torch.cat([l(x) for l in self.multi_scale_conv],1))
        z2 = self.reconstructor[0](z1)
        # z2 = self.reconstructor[1](z1)
        r = self.reconstructor[1](torch.cat((z1,z2), 1))
        return r
    
    def _get_feats(self, x, return_interm_activations=False, return_reconstruction=False, **kwargs):
        xp = self.preprocess(x)
        if self.params.no_reconstruction:
            r = xp
        else:
            r = self.reconstruct(xp)
        if self.params.use_residual_during_inference and (not self.training):
            x = (x - r)
        # plt.subplot(1,2,1)
        # plt.imshow(convert_image_tensor_to_ndarray(x[0]))
        # plt.subplot(1,2,2)
        # plt.imshow(convert_image_tensor_to_ndarray(r[0]))
        # plt.savefig('recon2.png')
        
        out = self._get_feats_(r, return_interm_activations=return_interm_activations, **kwargs)
        if return_reconstruction:
            if isinstance(out, tuple):
                out = out + (r,)
            else:
                out = (out, r)
        return out
    
    def forward_and_reconstruct(self, x):
        f, r = self._get_feats(x, return_reconstruction=True)
        if self.params.cls_wt > 0:
            logits = self._run_classifier(f)
        else:
            logits = torch.zeros((x.shape[0], self.params.num_classes), dtype=x.dtype, device=x.device)
        return logits, r

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        return out

class DnCNN(nn.Module):
    def __init__(self, num_channels, num_resblocks):
        super().__init__()
        # define the layers of the DnCNN
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_resblocks)]
        )
        self.conv2 = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # apply the convolutional layers and ReLU activation function
        x = torch.relu(self.conv1(x))
        x = self.resblocks(x)
        x = self.conv2(x)
        return x

class XResNetClassifierWithDeepResidualEnhancer(XResNetClassifierWithEnhancer):
    @define(slots=False)
    class ModelParams(XResNetClassifierWithEnhancer.ModelParams):
        perceptual_loss: bool = False
        ploss_cnn_layer_idx: int = 12
        resnet_ckp_path: str = None
        reconstructor_ckp_path: str = None

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        if params.resnet_ckp_path is not None:
            load_params_into_model(torch.load(params.resnet_ckp_path), self, r'(.*convs.*|.*multi_scale_conv.*|.*bnrelu.*|.*upsample.*)')
        if params.reconstructor_ckp_path is not None:
            load_params_into_model(torch.load(params.reconstructor_ckp_path), self, r'(.*resnet.*)')

    def _make_reconstructor(self):
        bnrelu = lambda c : nn.Sequential(nn.BatchNorm2d(c),nn.ReLU())
        convbnrelu = lambda ci, co, k, s, p: nn.Sequential(nn.Conv2d(ci, co, k, s, p),bnrelu(co))
    #     self.reconstructor = nn.Sequential(
    #         convbnrelu(3, 128, 15, 2, 7),
    #         convbnrelu(128, 320, 1, 1, 0),
    #         convbnrelu(320, 320, 1, 1, 0),
    #         convbnrelu(320, 320, 3, 2, 1),
    #         convbnrelu(320, 128, 1, 1, 0),
    #         convbnrelu(128, 128, 3, 1, 1),
    #         convbnrelu(128, 512, 1, 1, 0),
    #         convbnrelu(512, 48*4, 5, 1, 2),
    #         nn.PixelShuffle(2),
    #         convbnrelu(48, 96, 3, 1, 1),
    #         nn.Conv2d(96, 3*4, 5, 1, 2),
    #         nn.PixelShuffle(2)
    #     )
    
    # def reconstruct(self, x):
    #     return self.reconstructor(x)

        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(3, 256, 13, 2, 6),
            # nn.Conv2d(3, 64, 9, 2, 4),
            # nn.Conv2d(3, 64, 7, 2, 3),
            # nn.Conv2d(3, 64, 5, 2, 2),
        ])
        self.bnrelu = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.convs = nn.Sequential(
            convbnrelu(256, 64, 3, 1, 1),
            convbnrelu(64, 64, 3, 1, 1),
            *(8*[convbnrelu(128, 64, 3, 1, 1)]),
        )
        self.upsample = nn.Sequential(
            convbnrelu(128, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            convbnrelu(32, 32, 3, 1, 1),
            # nn.PixelShuffle(2),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
        # self.reconstructor = DnCNN(3, 16)
        if self.params.perceptual_loss:
            self.ploss_cnn = torchvision.models.vgg16_bn(True).features[:self.params.ploss_cnn_layer_idx+1].requires_grad_(False)

    def reconstruct(self, xo):
        x = self.bnrelu(torch.cat([l(xo) for l in self.multi_scale_conv],1))
        for i,l in enumerate(self.convs):
            f1 = l(x)
            if i > 0:
                x = torch.cat([f0,f1], 1)
            else:
                x = f1
            f0 = f1
        x = self.upsample(x)
        return x

    def compute_perceptual_loss(self, x, r):
        x = self.normalization_layer(x)
        r = self.normalization_layer(r)
        
        fx = self.ploss_cnn(x)
        fr = self.ploss_cnn(r)
        loss = torch.pow(fx - fr, 2).sum(1).mean()
        return loss

    def compute_loss(self, x, y, return_logits=True):
        if (self.params.recon_wt > 0):# and self.training:
            logits, recon = self.forward_and_reconstruct(x)
            if self.params.perceptual_loss:
                rloss = self.compute_perceptual_loss(x, recon)
            else:
                rloss = torch.pow(x.reshape(x.shape[0],-1) - recon.reshape(recon.shape[0],-1), 2).mean()
        else:
            logits = self.forward(x)
            rloss = 0
        closs = nn.functional.cross_entropy(logits, y)
        loss = self.params.cls_wt * closs + self.params.recon_wt * rloss
        if return_logits:
            return logits, loss
        return loss