from copy import deepcopy
from typing import List, Tuple, Union
import torch
from rblur.models import \
    CommonModelParams, CommonModelMixin
from attrs import define, field
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from torch import nn
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange

from rblur.models import ConvParams, IdentityLayer

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
        self.layer = nn.Linear(input_size, self.params.common_params.num_units, bias=self.params.common_params.bias)
    
    def forward(self, x, **kwargs):
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
        z: torch.Tensor = x
        z = z.transpose(1,2)
        z = self.spatial_mlp(z)
        z = z.transpose(1,2)
        x = x + z
        z = self.layernorm2(x)
        z = x + self.channel_mlp(z)
        z = self.layernorm1(z)
        return z

class MLPMixer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        patch_gen_params: BaseParameters = None
        mixer_block_params: List[MixerBlock.ModelParams] = field(factory=list)
        normalization_layer_params: BaseParameters = None
        classifier_params: BaseParameters = None
        use_cls_token: bool = False
        normalize_input: bool = True
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()

    def _make_network(self):
        self.patch_gen = self.params.patch_gen_params.cls(self.params.patch_gen_params)

        x = torch.rand(1, *(self.params.common_params.input_size))
        x = self.patch_gen(x)
        if isinstance(x, tuple):
            x = x[0]
        x = self._reshape_grid_to_patches(x)
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
        if self.params.use_cls_token:
            self.cls_token = nn.parameter.Parameter(torch.empty(1,1,x.shape[-1]).zero_())
        self.classifier = self.params.classifier_params.cls(self.params.classifier_params)
        self.layernorm = nn.LayerNorm(x.shape[-1])
        if self.params.normalize_input:
            self.normalization_layer = self.params.normalization_layer_params.cls(self.params.normalization_layer_params)
    
    def _reshape_grid_to_patches(self, x):
        if x.dim() == 4:
            x = rearrange(x, 'b c h w -> b (h w) c')
            # b, c, h, w = x.shape
            # x = x.transpose(1,2).transpose(2,3)
            # x = x.reshape(b,h*w,c)
        return x

    def _get_patch_emb(self, x):
        if self.params.normalize_input:
            x = self.normalization_layer(x)
        x: torch.Tensor = self.patch_gen(x)
        if isinstance(x, tuple):
            x = x[0]
        z = self._reshape_grid_to_patches(x)
        return z
    
    def _run_mixer_blocks(self, z):
        z = self.layernorm(z)
        z = self.mixer_blocks(z)
        return z
    
    def _get_feats(self, x):
        z = self._get_patch_emb(x)
        z = self._run_mixer_blocks(z)
        z = z.mean(1)
        return z

    def _run_classifier(self, z):
        y = self.classifier(z)
        return y

    def forward(self, x):
        z = self._get_feats(x)
        y = self._run_classifier(z)
        return y

    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        if return_logits:
            return logits, loss
        else:
            return loss

class UnfoldPatchExtractor(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        conv_params: ConvParams = field(factory=ConvParams)

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()
    
    def _make_network(self):
        self.unfold = nn.Unfold(self.params.conv_params.kernel_sizes[0],
                                padding=self.params.conv_params.padding[0],
                                stride=self.params.conv_params.strides[0])

    def forward(self, x):
        return self.unfold(x).transpose(1,2)

class FirstNExtractionClassifier(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        num_classes: int = None

    def forward(self, x):
        return x[:, :self.params.num_classes]