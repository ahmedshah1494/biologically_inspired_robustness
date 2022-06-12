import time
from turtle import forward
from typing import List, Type, Union
from attr import define
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
import numpy as np

from torch import dropout, nn
import torch

from model_utils import _make_first_dim_last, _make_last_dim_first, merge_strings, mm, str_to_act_and_dact_fn, _compute_conv_output_shape

@define(slots=False)
class CommonModelParams:
    input_size: Union[int, List[int]] = 0
    num_units: Union[int, List[int]] = 0
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
    no_act_after_update: bool = False

    max_train_time_steps: int = 1
    max_test_time_steps: int = 1

    legacy_act_update_normalization: bool = False
    sparsify_act: bool = False
    sparsity_coeff: float = 1.

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
        t0 = time.time()
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
        # print(step_idx, 'loss =', loss.detach().cpu().numpy().mean(), f'max update norm = {act_update_norm.max():.3e} (/{scale.min():.3f})', act_update_m.shape, scale.shape, (state>0).float().mean().cpu().detach().numpy(), time.time() - t0)
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

class ConsistentActivationLayer(AbstractModel, CommonModelMixin, ConsistencyOptimizationMixin):
    class ModelParams(BaseParameters):
        def __init__(self, cls):
            super().__init__(cls)
            self.common_params: CommonModelParams = CommonModelParams()
            self.consistency_optimization_params: ConsistencyOptimizationParams = ConsistencyOptimizationParams()        

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
    class ModelParams(ConsistentActivationLayer.ModelParams):
        def __init__(self, cls):
            super().__init__(cls)
            self.classification_params: ClassificationModelParams = ClassificationModelParams()   

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
    class ModelParams(BaseParameters):
        def __init__(self, cls):
            super().__init__(cls)
            self.common_params: CommonModelParams = CommonModelParams()
            self.consistency_optimization_params: ConsistencyOptimizationParams = ConsistencyOptimizationParams()
            self.scanning_consistency_optimization_params: ScanningConsistencyOptimizationParams = ScanningConsistencyOptimizationParams()

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

    def _optimize_activations(self, state_hist, x, W_lat, b_lat, W_back, b_back):
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
        super()._optimize_activations(state_hist, x, W_lat, b_lat, W_back, b_back)

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
        def reshape_state(s):
            if self.input_act_consistency and (not self.window_input_act_consistency):
                s = _make_last_dim_first(s)
            else:
                s = s.transpose(1,2).reshape(s.shape[0], -1)
                s = self._fold_for_act_opt(s, _make_first_dim_last(fwd).shape)
                s = s.reshape(s.shape[0], s.shape[1], *(self.output_spatial_dims))
            return s
        logits = reshape_state(state_hist[-1])
        if return_state_hist:
            state_hist = [reshape_state(s) for s in state_hist]
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

class SequentialLayers(AbstractModel, CommonModelMixin):
    class ModelParams(BaseParameters):
        def __init__(self, cls):
            super().__init__(cls)
            self.layer_params: List[BaseParameters] = []
            self.common_params: CommonModelParams = CommonModelParams()
    
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
        self.activation = self.params.common_params.activation()
        self.dropout = nn.Dropout(self.params.common_params.dropout_p)

        x = torch.rand(1,*(self.params.common_params.input_size))
        layers = []
        print(x.shape)
        for lp in self.params.layer_params:
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
            out = self.activation(out)
            out = self.dropout(out)
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

class GeneralClassifier(AbstractModel):
    class ModelParams(BaseParameters):
        feature_model_params: BaseParameters
        classifier_params: BaseParameters
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self._make_network()
        self._make_name()
    
    def _make_name(self):
        self.name = self.feature_model.name
    
    def _make_network(self):
        fe_params = self.params.feature_model_params
        self.feature_model = fe_params.cls(fe_params)

        x = torch.rand(1,*(fe_params.common_params.input_size))
        x = self.feature_model(x)
        if isinstance(x, tuple):
            x = x[0]

        cls_params = self.params.classifier_params
        cls_params.common_params.input_size = x.shape[1:]
        self.classifier = cls_params.cls(cls_params)
    
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
        if np.random.uniform(0, 1) < 1/100:
            _r = r.detach().cpu().numpy()
            print(f'sparsity={(_r == 0).astype(float).mean():3f}', _r.max(), np.quantile(_r, 0.99))
        out = self.classifier(r, *fwd_args, **fwd_kwargs)
        if isinstance(out, tuple):
            logits = out[0]
            cls_extra_outputs = out[1:]
            extra_outputs.append(cls_extra_outputs)
        else:
            logits = out
            cls_extra_outputs = ()
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
        if (logits.dim() == 1) or ((logits.dim() > 1) and (logits.shape[-1] == 1)):
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        else:
            loss = nn.functional.cross_entropy(logits, y)
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
    class ModelParams(BaseParameters):
        common_params: CommonModelParams
        photoreceptor_params: BaseParameters
        horizontal_cell_params: BaseParameters
        bipolar_cell_params: BaseParameters
        amacrine_cell_params: BaseParameters
        ganglion_cell_params: BaseParameters
    
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
        out = maybe_acitvate(self.photoreceptors, out)

        out = forward_and_maybe_loss(self.h_cells, out)
        out = maybe_acitvate(self.h_cells, out)

        out = forward_and_maybe_loss(self.bp_cells, out)
        out = maybe_acitvate(self.bp_cells, out)
        out = maybe_dropout(self.bp_cells, out)
        
        a_proj = forward_and_maybe_loss(self.a_cells, out)
        a_proj = maybe_acitvate(self.a_cells, a_proj)
        a_proj = maybe_dropout(self.a_cells, a_proj)
        
        out = forward_and_maybe_loss(self.g_cells, a_proj + out)
        out = maybe_acitvate(self.g_cells, out)
        out = maybe_dropout(self.g_cells, out)
        
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
        common_params: CommonModelParams = None
        conv_params: ConvParams = None
        def __attrs_post_init__(self):
            self.common_params: CommonModelParams = CommonModelParams()
            self.conv_params: ConvParams = ConvParams()            

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