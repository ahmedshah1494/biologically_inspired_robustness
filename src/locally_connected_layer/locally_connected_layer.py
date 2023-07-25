import torch
from torch import nn
import numpy as np
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from torch.nn.modules.utils import _pair
from attrs import define

npa = np.array
class _LocallyConnectedLayer(nn.Module):
    '''
    '''
    def __init__(self, kernel_size, in_channels, out_channels, input_size, stride=1, padding=0, dilation=1, bias=True) -> None:
        super().__init__()
        input_size = _pair(input_size)

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.unfold = nn.Unfold(self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation)
        # x = torch.rand(1, in_channels, *input_size)
        # xunfold = self.unfold(x)
        # _, M, L = xunfold.shape

        self.output_shape = np.floor(1 + (npa(input_size) + 2*npa(self.padding) - npa(self.dilation)*(npa(self.kernel_size) - 1) - 1) / npa(self.stride)).astype(int)
        M = np.prod(self.kernel_size)*in_channels
        L = np.prod(self.output_shape)

        # self.fold = nn.Fold(input_size, kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation)

        self.weight = nn.parameter.Parameter(nn.init.xavier_uniform_(torch.empty(1, L, M, out_channels)), requires_grad=True)
        if bias:
            self.bias = nn.parameter.Parameter(nn.init.xavier_uniform_(torch.empty(1,out_channels,1,1)), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x):
        # print(x.shape)
        x = self.unfold(x) # B x M x L
        # print(x.shape)
        x = x.transpose(2,1).unsqueeze(2) # B x L x 1 x M
        # print(x.shape)
        x = torch.matmul(x, self.weight)  # B x L x 1 x C_out
        # print(x.shape)
        #   B x L x C_out    B x C_out x L
        x = x.squeeze(2).transpose(2,1)
        # print(x.shape)
        x = x.reshape(x.shape[0], x.shape[1], *self.output_shape)
        # print(x.shape)
        return x
    
class LocallyConnectedLayer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        kernel_size: int = None
        in_channels: int = None
        out_channels: int = None
        input_size: int = None
        stride: int = 1
        padding: int = 0
        dilation: int = 1
        bias: bool = True
    
    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self._make_network()
    
    def _make_network(self):
        kwargs_to_exclude = set(['cls'])
        kwargs = self.params.asdict(filter=lambda a,v: a.name not in kwargs_to_exclude)
        self.lcl = _LocallyConnectedLayer(**kwargs)

    def forward(self, x):
        return self.lcl(x)

    def compute_loss(self, x, y, return_logits=True):
        out = self.forward(x)
        logits = out
        loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss