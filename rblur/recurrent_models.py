from copy import deepcopy
from turtle import forward
from typing import List, Tuple, Type, Union
import numpy as np

import torch
from rblur.models import \
    CommonModelParams, CommonModelMixin
from attrs import define, field
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from torch import nn

def detach_list_of_tensors(l: List[torch.Tensor]):
    return [x.detach() if isinstance(x, torch.Tensor) else x for x in l]

def detach_and_move_to_cpu(l: List[torch.Tensor]):
    return [x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in l]

class BaseRecurrentCell(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        # Parameters for the update networks. The output of the update networks are
        # added to the hidden state in each time-step
        forward_update_params: BaseParameters = None
        lateral_update_params: BaseParameters = None
        backward_update_params: BaseParameters = None

        # Parameters for upscaling the hidden state (either pre-act or post-act) to
        # match the size of the input.
        backward_act_params: BaseParameters = None
        hidden_backward_params: BaseParameters = None

        # Parameters for the combiner layers used to combine the inputs of the update networks.
        forward_input_combiner_params: BaseParameters = None
        lateral_input_combiner_params: BaseParameters = None
        backward_input_combiner_params: BaseParameters = None

        use_layernorm: bool = True
        common_params: CommonModelParams = field(factory=CommonModelParams)
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()

    def _init_combiners(self):
        self.forward_input_combiner: AbstractInputCombinationLayer = self.params.forward_input_combiner_params.cls(self.params.forward_input_combiner_params)
        self.lateral_input_combiner: AbstractInputCombinationLayer = self.params.lateral_input_combiner_params.cls(self.params.lateral_input_combiner_params)
        self.backward_input_combiner: AbstractInputCombinationLayer = self.params.backward_input_combiner_params.cls(self.params.backward_input_combiner_params)

    def _init_update_nets(self):
        # We need to set the input size for the forward_input_combiner here to take into account the effect of the input combiner.
        # This is required especially in the case of convolutional networks because we pass a dummy input to estimate the output shape.
        self.params.forward_update_params.common_params.input_size = list(self.params.common_params.input_size)[:]
        self.params.forward_update_params.common_params.input_size[self.forward_input_combiner.params.combiner_params.dim - 1] *= self.forward_input_combiner.mul_dim_factor
        self.params.forward_update_params.common_params.input_size[self.forward_input_combiner.params.combiner_params.dim - 1] += self.forward_input_combiner.add_dim_factor
        self.forward_update_net = self.params.forward_update_params.cls(self.params.forward_update_params)
        self.lateral_update_net = self.params.lateral_update_params.cls(self.params.lateral_update_params)
        self.backward_update_net = self.params.backward_update_params.cls(self.params.backward_update_params)

    def _init_upscaler(self, p, input_shape, output_shape):
        if p is not None:
            if isinstance(p, (ConvTransposeUpscaler.ModelParams, LinearUpscaler.ModelParams)):
                p.target_shape = input_shape
            p.common_params.input_shape = output_shape
            upscaler = p.cls(p)
            return upscaler

    def _init_backward_nets(self):
        self.backward_act_net = self._init_upscaler(self.params.backward_act_params, self.params.common_params.input_size, self.hidden_size)
        self.hidden_backward_net = self._init_upscaler(self.params.hidden_backward_params, self.params.common_params.input_size, self.hidden_size)
        
        if (self.params.backward_act_params is not None) and (self.params.hidden_backward_params is None):
            self.hidden_backward_net = self.backward_act_net
        if (self.params.backward_act_params is None) and (self.params.hidden_backward_params is not None):
            self.backward_act_net = self.hidden_backward_net

    def _make_network(self):
        self._init_combiners()
        self._init_update_nets()

        self.hidden_size = self.forward_update_net(torch.rand(1,*(self.params.forward_update_params.common_params.input_size))).shape[1:]
        self.init_hidden = nn.parameter.Parameter(torch.empty(*self.hidden_size).zero_(), requires_grad=False)
        self.init_feedback = nn.parameter.Parameter(torch.empty(*self.hidden_size).zero_(), requires_grad=False)

        self._init_backward_nets()

        if self.params.use_layernorm:
            self.layernorm = nn.LayerNorm(self.hidden_size)

        self.activation = self.params.common_params.activation()
    
    def _make_name(self):
        self.name = 'BaseRecurrentCell'

    def _combine_inputs_for_update(self, x, h, a, a_b):
        h_b = self.hidden_backward_net(h)
        if hasattr(self, 'forward_input_combiner'):
            fwdinp = self.forward_input_combiner(x, h_b)
            latinp = self.lateral_input_combiner(a, h)
            backinp = self.backward_input_combiner(a_b, h)
        else:
            fwdinp = torch.cat((x, h_b), dim=1)
            latinp = torch.cat((a, h), dim=1)
            backinp = torch.cat((a_b, h), dim=1)

        return fwdinp, latinp, backinp

    def forward(self, x: torch.Tensor, h:torch.Tensor, a:torch.Tensor, a_b:torch.Tensor, return_hidden=True, return_backward_act=True):
        if h is None:
            h = self.init_hidden
            h = h.unsqueeze(0)
            h = torch.repeat_interleave(h, x.shape[0], dim=0)
            # if self.params.use_layernorm:
            #     h = self.layernorm(h)

        if a is None:
            a = self.activation(h)

        if a_b is None:
            a_b = self.activation(self.init_feedback)
            a_b = a_b.unsqueeze(0)
            a_b = torch.repeat_interleave(a_b, h.shape[0], dim=0)

        fwdinp, latinp, backinp = self._combine_inputs_for_update(x, h, a, a_b)
        u_f = self.forward_update_net(fwdinp)
        u_l = self.lateral_update_net(latinp)
        u_b = self.backward_update_net(backinp)
        update = u_f + u_l + u_b
        h_new = h + u_f + u_l + u_b
        if self.params.use_layernorm:
            h_new = self.layernorm(h_new)

        a_new = self.activation(h_new)
        
        outputs = (a_new,)
        if return_hidden:
            outputs = outputs + (h_new,)
        if return_backward_act:
            a_up = self.activation(self.backward_act_net(a_new))
            outputs = outputs + (a_up,)
        outputs = outputs if (len(outputs) > 1) else outputs[0]
        return outputs

@define(slots=False)
class RecurrenceParams:
    train_time_steps: int = 1
    test_time_steps: int = 1
    truncated_loss_wt: float = 0

class RecurrentModel(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        cell_params: List[BaseRecurrentCell.ModelParams] = None
        recurrence_params: RecurrenceParams = field(factory=RecurrenceParams)

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()

    def _make_network(self):
        cells = []
        isize = self.params.common_params.input_size
        for cp in self.params.cell_params:
            cp.common_params.input_size = isize
            cell: BaseRecurrentCell = cp.cls(cp)
            cells.append(cell)
            x = torch.rand(1, *(isize))
            isize = cell(x, None, None, None, return_hidden=False, return_backward_act=False).shape[1:]
        self.cells = nn.ModuleList(cells)

    def _single_step(self, inp, hidden_states, activations, feedbacks):
        new_hidden_states = []
        new_activations = []
        new_feedbacks = []
        for i,(h, a, ab, cell) in enumerate(zip(hidden_states, activations, feedbacks, self.cells)):
            outputs = cell(inp, h, a, ab, return_backward_act=(i > 0))
            inp = outputs[0]
            new_activations.append(outputs[0])
            new_hidden_states.append(outputs[1])
            if i > 0:
                new_feedbacks.append(outputs[2])
        new_feedbacks.append(None)
        return new_hidden_states, new_activations, new_feedbacks

    def _run_all_steps(self, num_time_steps, inp, hidden_states, activations, feedbacks):
        all_hidden_states = [hidden_states]
        all_activations = [activations]
        all_feedbacks = [feedbacks]
        all_logits = []

        for t in range(num_time_steps):
            hidden_states, activations, feedbacks = self._single_step(inp, hidden_states, activations, feedbacks)
            all_logits.append(hidden_states[-1])
            all_hidden_states.append(hidden_states)
            all_activations.append(activations)
            all_feedbacks.append(feedbacks)
        return all_logits, all_hidden_states, all_activations, all_feedbacks

    def forward(self, x, return_all_logits=False, return_hidden_acts=False, return_hidden_states=False, return_feedback_acts=False):        
        hidden_states = activations = feedbacks = [None]*len(self.cells)
        T = self.params.recurrence_params.train_time_steps if self.training else self.params.recurrence_params.test_time_steps
        all_logits, all_hidden_states, all_activations, all_feedbacks = self._run_all_steps(T, x, hidden_states, activations, feedbacks)
        
        outputs = (all_logits[-1],)
        if return_all_logits:
            outputs += (all_logits,)
        if return_hidden_acts:
            outputs += (all_activations,)
        if return_hidden_states:
            outputs += (all_hidden_states,)
        if return_feedback_acts:
            outputs += (all_feedbacks,)
            
        if len(outputs) == 1:
            outputs = outputs[0] 
        return outputs
    
    def compute_loss(self, x, y, return_logits=True):
        # final_logits, all_logits, all_activations, all_hidden_states, all_feedbacks = self.forward(x, return_all_logits=True, return_hidden_acts=True, return_hidden_states=True, return_feedback_acts=True)
        T = self.params.recurrence_params.train_time_steps if self.training else self.params.recurrence_params.test_time_steps
        alpha = self.params.recurrence_params.truncated_loss_wt

        hidden_states = activations = feedbacks = [None]*len(self.cells)
        all_logits, all_hidden_states, all_activations, all_feedbacks = self._run_all_steps(T, x, hidden_states, activations, feedbacks)
        full_pass_loss = nn.functional.cross_entropy(all_logits[-1], y) #sum([nn.functional.cross_entropy(logits, y) for logits in all_logits])

        if alpha > 0 and self.training:        
            n = np.random.randint(0, T)
            k = np.random.randint(1, T-n+1)
            trunc_all_logits = self._run_all_steps(k, x, detach_list_of_tensors(all_hidden_states[n]), detach_list_of_tensors(all_activations[n]), detach_list_of_tensors(all_feedbacks[n]))[0]
            trunc_pass_loss = nn.functional.cross_entropy(trunc_all_logits[-1], y) #sum([nn.functional.cross_entropy(logits, y) for logits in trunc_all_logits])
            loss = (1-alpha) * full_pass_loss + alpha*trunc_pass_loss
        else:
            loss = full_pass_loss        
        logits = all_logits[-1]
        if np.random.rand() < 1/100:
            acc_ovr_time = [float((torch.argmax(l, dim=1) == y).float().mean().detach().cpu()) for l in all_logits]
            if self.training:
                trunc_acc_ovr_time = [float((torch.argmax(l, dim=1) == y).float().mean().detach().cpu()) for l in trunc_all_logits]
            else:
                trunc_acc_ovr_time = []
            print(acc_ovr_time, trunc_acc_ovr_time)

        if return_logits:
            return logits, loss
        else:
            return loss 

class InputCombinationParams:
    dim: int = -1
    num_inputs:int = None

class AbstractInputCombinationLayer(AbstractModel):
    add_dim_factor = 0
    mul_dim_factor = 1

    @define(slots=False)
    class ModelParams(BaseParameters):
        combiner_params: InputCombinationParams = field(factory=InputCombinationParams)
    
    def _preprocess_inputs(self, *inputs):
        return inputs

class InputConcatenationLayer(AbstractInputCombinationLayer):    
    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self.add_dim_factor = 0
        self.mul_dim_factor = self.params.combiner_params.num_inputs

    def forward(self, *inputs):
        inputs = self._preprocess_inputs(*inputs)
        return torch.cat(inputs, dim=self.params.combiner_params.dim)

class Linear(AbstractModel, CommonModelMixin):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
    
    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self.load_common_params()
        self._make_network()
    
    def _make_network(self):
        indim = self.input_size
        if np.iterable(indim):
            indim = np.prod(indim)
        self.dense = nn.Linear(indim, self.num_units, self.use_bias)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.dense(x)

@define(slots=True)
class ConvParams:
    in_channels: int = None
    out_channels: int = None
    kernel_size: int = None
    stride: int = 1
    padding: int = 0
    dilation: int = 1

class ConvTransposeUpscaler(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=lambda: CommonModelParams(bias=False))
        conv_params: ConvParams = field(factory=ConvParams)
        target_shape: Union[torch.Size, Tuple[int]] = None
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()
    
    def _make_network(self):
        ic = self.params.conv_params.out_channels
        oc = self.params.target_shape[0]
        k = self.params.conv_params.kernel_size
        s = self.params.conv_params.stride
        p = self.params.conv_params.padding
        d_in = np.array(self.params.common_params.input_shape[-2:])
        d_out = np.array(self.params.target_shape[-2:])
        op = d_out - ((d_in - 1) * np.array(s) - 2 * np.array(p) + np.array(k)-1 + 1)
        self.upsample = nn.ConvTranspose2d(ic, oc, k, s, p, tuple(op), bias=self.params.common_params.bias)
    
    def forward(self, x):
        return self.upsample(x)

class LinearUpscaler(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=lambda: CommonModelParams(bias=False))
        target_shape: Union[torch.Size, Tuple[int]] = None

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()
    
    def _make_network(self):
        indim = self.params.common_params.input_shape
        if np.iterable(indim):
            indim = np.prod(indim)
        outdim = self.params.target_shape
        if np.iterable(outdim):
            outdim = np.prod(outdim)
        self.dense = nn.Linear(indim, outdim, bias=self.params.common_params.bias)
    
    def forward(self, x):
        out = self.dense(x.reshape(x.shape[0], -1))
        out = out.reshape(out.shape[0], *list(self.params.target_shape))
        return out

class Conv2d(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        conv_params: ConvParams = field(factory=ConvParams)
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()
    
    def _make_network(self):
        if self.params.common_params.input_size is not None:
            ic = self.params.common_params.input_size[0]
        else:
            ic = self.params.conv_params.in_channels
        self.conv = nn.Conv2d(ic, self.params.conv_params.out_channels, 
                                self.params.conv_params.kernel_size, self.params.conv_params.stride, 
                                self.params.conv_params.padding, self.params.conv_params.dilation)

    def forward(self, x, return_upscaled=False):
        return self.conv(x)

class Conv2DSelfProjection(AbstractModel):
    @define(slots=True)
    class ModelParams(BaseParameters):
        conv_params: ConvParams = field(factory=ConvParams)
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()
    
    def _make_network(self):
        # out_channels = self.params.conv_params.in_channels * (self.params.conv_params.kernel_size ** 2)
        self.conv = nn.Conv2d(self.params.conv_params.in_channels, self.params.conv_params.out_channels,
                                self.params.conv_params.kernel_size, self.params.conv_params.stride,
                                bias=False)
    
    def forward(self, x):
        bs, ic, h, w = x.shape
        proj = self.conv(x)
        bs, oc, _, _ = proj.shape
        proj_reshaped = torch.nn.functional.fold(proj.reshape(bs, oc, -1), (h,w), 
                                    self.params.conv_params.kernel_size, 
                                    stride=self.params.conv_params.stride)
        return proj_reshaped

class UpscaleAndCombine(AbstractInputCombinationLayer):
    @define(slots=False)
    class ModelParams(BaseParameters):
        combiner_params: AbstractInputCombinationLayer.ModelParams = None
        upscaler_params: BaseParameters = None
    
    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self._make_network()
    
    def _make_network(self):
        self.upscaler = self.params.upscaler_params.cls(self.params.upscaler_params)
        self.combiner = self.params.combiner_params.cls(self.params.combiner_params)
    
    def forward(self, a, b):
        """
            Upscales b to match the shape of a.
        """
        bup = self.upscaler(b)
        comb = self.combiner(a,b)
        return comb
