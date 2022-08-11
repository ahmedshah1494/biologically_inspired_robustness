from turtle import forward
from typing import List, Tuple
import torch
from adversarialML.biologically_inspired_models.src.models import \
    CommonModelParams, CommonModelMixin
from attrs import define, field
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from torch import nn
import numpy as np

class LayerNormLayer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        dim_range: Tuple[int] = None
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self.ln = nn.LayerNorm(self.params.common_params.input_size[self.params.dim_range[0]: self.params.dim_range[1]])
    
    def forward(self, x):
        return self.ln(x)

class LinearLayer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()

    def _make_network(self):
        input_size = self.params.common_params.input_size
        if np.iterable(input_size):
            input_size = np.prod(input_size)
        self.layer = nn.Linear(input_size, self.params.common_params.num_units)
    
    def forward(self, x):
        return self.layer(x)

class MixerMLP(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()

    def _make_network(self):
        input_size = self.params.common_params.input_size
        if np.iterable(input_size):
            input_size = np.prod(input_size)
        hidden_size = self.params.common_params.num_units
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            self.params.common_params.activation(),
            nn.Dropout(self.params.common_params.dropout_p),
            nn.Linear(hidden_size, input_size),
        )
    
    def forward(self, x):
        return self.mlp(x)

class MixerBlock(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        channel_mlp_params: BaseParameters = None
        spatial_mlp_params: BaseParameters = None
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()

    def _make_network(self):
        self.channel_mlp = self.params.channel_mlp_params.cls(self.params.channel_mlp_params)
        self.spatial_mlp = self.params.spatial_mlp_params.cls(self.params.spatial_mlp_params)
        self.layernorm1 = nn.LayerNorm(self.params.common_params.input_size[-1])
        self.layernorm2 = nn.LayerNorm(self.params.common_params.input_size[-1])
    
    def forward(self, x: torch.Tensor):
        z: torch.Tensor = self.layernorm1(x)
        z = z.transpose(1,2)
        z = self.spatial_mlp(z)
        z = z.transpose(1,2)
        z = self.layernorm2(x + z)
        z = self.channel_mlp(z)
        return x + z

class MLPMixer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        patch_gen_params: BaseParameters = None
        mixer_block_params: List[MixerBlock.ModelParams] = field(factory=list)
        classifier_params: BaseParameters = None
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()

    def _make_network(self):
        self.patch_gen = self.params.patch_gen_params.cls(self.params.patch_gen_params)

        x = torch.rand(1, *(self.params.common_params.input_size))
        x = self._reshape_grid_to_patches(self.patch_gen(x))
        mixer_blocks = []
        for block_p in self.params.mixer_block_params:
            block_p.common_params.input_size = x.shape[1:]
            x = torch.rand(1, *(x.shape[1:]))
            block = block_p.cls(block_p)
            x = block(x)
            mixer_blocks.append(block)
        self.mixer_blocks = nn.Sequential(
            *mixer_blocks
        )
        self.classifier = self.params.classifier_params.cls(self.params.classifier_params)
    
    def _reshape_grid_to_patches(self, x):
        b, c, h, w = x.shape
        z = x.transpose(1,2).transpose(2,3)
        z = z.reshape(b,h*w,c)
        return z
    
    def forward(self, x):
        x: torch.Tensor = self.patch_gen(x)
        z = self._reshape_grid_to_patches(x)
        z = self.mixer_blocks(z)
        z = z.mean(1)
        y = self.classifier(z)
        return y
    
    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        if return_logits:
            return logits, loss
        else:
            return loss
    
