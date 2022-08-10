from copy import deepcopy
from turtle import forward
from typing import List, Tuple, Type, Union
import numpy as np

import torch
from adversarialML.biologically_inspired_models.src.models import \
    CommonModelParams, CommonModelMixin
from attrs import define, field
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from torch import nn

class BaseRecurrentCell(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        forward_update_params: BaseParameters = None
        lateral_update_params: BaseParameters = None
        backward_update_params: BaseParameters = None
        backward_act_params: BaseParameters = None
        hidden_backward_params: BaseParameters = None
        use_layernorm: bool = True
        common_params: CommonModelParams = field(factory=CommonModelParams)
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()

    def _make_network(self):
        self.params.forward_update_params.common_params.input_size = list(self.params.common_params.input_size)[:]
        self.params.forward_update_params.common_params.input_size[0] *= 2
        self.forward_update_net = self.params.forward_update_params.cls(self.params.forward_update_params)

        hidden_size = self.forward_update_net(torch.rand(1,*(self.params.forward_update_params.common_params.input_size))).shape[1:]
        self.init_hidden = nn.parameter.Parameter(torch.empty(*hidden_size).zero_(), requires_grad=False)
        self.init_feedback = nn.parameter.Parameter(torch.empty(*hidden_size).zero_(), requires_grad=False)

        if self.params.use_layernorm:
            self.layernorm = nn.LayerNorm(hidden_size)
        
        if self.params.backward_act_params is not None:
            if isinstance(self.params.backward_act_params, (ConvTransposeUpscaler.ModelParams, LinearUpscaler.ModelParams)):
                self.params.backward_act_params.target_shape = self.params.common_params.input_size
                self.params.backward_act_params.input_shape = hidden_size
            self.backward_act_net = self.params.backward_act_params.cls(self.params.backward_act_params)
        
        if self.params.hidden_backward_params is not None:
            if isinstance(self.params.hidden_backward_params, (ConvTransposeUpscaler.ModelParams, LinearUpscaler.ModelParams)):
                self.params.hidden_backward_params.target_shape = self.params.common_params.input_size
                self.params.hidden_backward_params.input_shape = hidden_size
            self.hidden_backward_net = self.params.hidden_backward_params.cls(self.params.hidden_backward_params)
        
        if (self.params.backward_act_params is not None) and (self.params.hidden_backward_params is None):
            self.hidden_backward_net = self.backward_act_net
        if (self.params.backward_act_params is None) and (self.params.hidden_backward_params is not None):
            self.backward_act_net = self.hidden_backward_net

        self.lateral_update_net = self.params.lateral_update_params.cls(self.params.lateral_update_params)
        self.backward_update_net = self.params.backward_update_params.cls(self.params.backward_update_params)
        self.activation = self.params.common_params.activation()
    
    def _make_name(self):
        self.name = 'BaseRecurrentCell'

    def combine_inputs(self, a, b):
        return torch.cat((a,b), 1)

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

        h_b = self.hidden_backward_net(h)
        u_f = self.forward_update_net(self.combine_inputs(x, h_b))
        u_l = self.lateral_update_net(self.combine_inputs(a, h))
        u_b = self.backward_update_net(self.combine_inputs(a_b, h))
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

def detach_and_move_to_cpu(l: List[torch.Tensor]):
    return [x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in l]

@define(slots=False)
class RecurrenceParams:
    num_time_steps: int = 1

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


    def forward(self, x):
        all_hidden_states = []
        all_activations = []
        all_feedbacks = []

        inp = x
        for t in range(self.params.recurrence_params.num_time_steps):
            if t == 0:
                hidden_states = activations = feedbacks = [None]*len(self.cells)
            hidden_states, activations, feedbacks = self._single_step(inp, hidden_states, activations, feedbacks)
            all_hidden_states.append(detach_and_move_to_cpu(hidden_states))
            all_activations.append(detach_and_move_to_cpu(activations))
            all_feedbacks.append(detach_and_move_to_cpu(feedbacks))        
        return activations[-1]
    
    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        if return_logits:
            return logits, loss
        else:
            return loss            

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
        conv_params: ConvParams = field(factory=ConvParams)
        target_shape: Union[torch.Size, Tuple[int]] = None
        input_shape: Union[torch.Size, Tuple[int]] = None
        use_bias: bool = False
    
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
        d_in = np.array(self.params.input_shape[-2:])
        d_out = np.array(self.params.target_shape[-2:])
        op = d_out - ((d_in - 1) * np.array(s) - 2 * np.array(p) + np.array(k)-1 + 1)
        self.upsample = nn.ConvTranspose2d(ic, oc, k, s, p, tuple(op), bias=self.params.use_bias)
    
    def forward(self, x):
        return self.upsample(x)

class LinearUpscaler(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        target_shape: Union[torch.Size, Tuple[int]] = None
        input_shape: Union[torch.Size, Tuple[int]] = None
        use_bias: bool = False
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()
    
    def _make_network(self):
        indim = self.params.input_shape
        if np.iterable(indim):
            indim = np.prod(indim)
        outdim = self.params.target_shape
        if np.iterable(outdim):
            outdim = np.prod(outdim)
        self.dense = nn.Linear(indim, outdim, bias=self.params.use_bias)
    
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
        self.conv = nn.Conv2d(self.params.common_params.input_size[0], self.params.conv_params.out_channels, 
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

