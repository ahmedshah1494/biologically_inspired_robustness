from copy import deepcopy
from typing import List, Tuple, Union
import torch
from adversarialML.biologically_inspired_models.src.models import \
    CommonModelParams, CommonModelMixin
from attrs import define, field
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from torch import nn
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange

from adversarialML.biologically_inspired_models.src.models import ConsistencyOptimizationMixin, ConsistencyOptimizationParams, ConsistentActivationLayer, ConvParams

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

class NormalizationLayer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        mean: Union[float, List[float]] = None
        std: Union[float, List[float]] = None

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        if isinstance(self.params.mean, list):
            self.mean = nn.parameter.Parameter(torch.FloatTensor(self.params.mean).reshape(1,-1,1,1), requires_grad=False)
        if isinstance(self.params.std, list):
            self.std = nn.parameter.Parameter(torch.FloatTensor(self.params.std).reshape(1,-1,1,1), requires_grad=False)
    
    def forward(self, x):
        return (x-self.mean)/self.std
    
    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = torch.zeros((logits.shape[0],), device=x.device)
        return logits
        # if return_logits:
        #     return logits, loss
        # else:
        #     return loss

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
        self.mean = nn.parameter.Parameter(torch.FloatTensor([0.4914, 0.4822, 0.4465]).reshape(1,3,1,1), requires_grad=False)
        self.std = nn.parameter.Parameter(torch.FloatTensor([0.2470, 0.2435, 0.2616]).reshape(1,3,1,1), requires_grad=False)
    
    def _reshape_grid_to_patches(self, x):
        if x.dim() == 4:
            x = rearrange(x, 'b c h w -> b (h w) c')
            # b, c, h, w = x.shape
            # x = x.transpose(1,2).transpose(2,3)
            # x = x.reshape(b,h*w,c)
        return x
    
    def forward(self, x):
        if self.params.normalize_input:
            x = (x - self.mean)/self.std
        x: torch.Tensor = self.patch_gen(x)
        if isinstance(x, tuple):
            x = x[0]
        z = self._reshape_grid_to_patches(x)
        z = self.layernorm(z)
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

class ConsistentActivationMixerMLP(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        consistency_optimization_params: ConsistencyOptimizationParams = field(factory=ConsistencyOptimizationParams)
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()

    def _make_network(self):
        def _get_ca_layer_params(indim, outdim):
            p: ConsistentActivationLayer.ModelParams = ConsistentActivationLayer.get_params()
            p.common_params.input_size = indim
            p.common_params.num_units = outdim
            p.common_params.bias = self.params.common_params.bias
            p.consistency_optimization_params = deepcopy(self.params.consistency_optimization_params)
            return p

        input_size = self.params.common_params.input_size
        if np.iterable(input_size):
            input_size = np.prod(input_size)
        hidden_size = self.params.common_params.num_units

        l1_params = _get_ca_layer_params(input_size, hidden_size)
        l1_params.common_params.activation = self.params.common_params.activation
        l1_params.common_params.dropout_p = self.params.common_params.dropout_p
        l1 = l1_params.cls(l1_params)

        # l2_params = _get_ca_layer_params(hidden_size, input_size)
        # l2_params.common_params.activation = nn.Identity
        # l2_params.consistency_optimization_params.lateral_dependence_type = 'Linear'
        # l2 = l2_params.cls(l2_params)
        l2_params: LinearLayer.ModelParams = LinearLayer.get_params()
        l2_params.common_params.input_size = hidden_size
        l2_params.common_params.num_units = input_size
        l2 = l2_params.cls(l2_params)

        self.mlp = nn.Sequential(l1, l2)
    
    def forward(self, x):
        # print(x.shape)
        b, n, c = x.shape
        x = x.reshape(b*n, c)
        y = self.mlp(x)
        y = y.reshape(b, n, c)
        return y

class ConsistentActivationMixerBlock(MixerBlock, ConsistencyOptimizationMixin):
    @define(slots=False)
    class ModelParams(MixerBlock.ModelParams):
        consistency_optimization_params: ConsistencyOptimizationParams = field(factory=ConsistencyOptimizationParams)
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.load_consistency_opt_params()
    
    def _make_network(self):
        super()._make_network()
        self.lat1 = nn.Linear(self.params.common_params.input_size[0], self.params.common_params.input_size[0])
        self.back1 = nn.Linear(self.params.common_params.input_size[0], self.params.common_params.input_size[0])
        self.lat2 = nn.Linear(self.params.common_params.input_size[1], self.params.common_params.input_size[1])
        self.back2 = nn.Linear(self.params.common_params.input_size[1], self.params.common_params.input_size[1])
        self.activation = nn.Identity()
    
    def _get_optimized_activations(self, z, x, lat_layer, back_layer):
        sh = [z.reshape(-1, z.shape[-1]).T]
        self._optimize_activations(sh, x.reshape(-1, x.shape[-1]).T, lat_layer.weight, lat_layer.bias.reshape(-1,1), back_layer.weight, back_layer.bias.reshape(-1,1))
        z = sh[-1].T.reshape(*(z.shape))
        return z

    def forward(self, x: torch.Tensor):
        z: torch.Tensor = x
        z = z.transpose(1,2)
        z = self.spatial_mlp(z)
        z = z.transpose(1,2)
        x_ = x + z
        z = self.layernorm2(x_)
        z = self._get_optimized_activations(z.transpose(1,2), x.transpose(1,2), self.lat1, self.back1).transpose(1,2)
        z = x_ + self.channel_mlp(z)
        z = self.layernorm1(z)
        z = self._get_optimized_activations(z, x, self.lat2, self.back2)
        return z

class FirstNExtractionClassifier(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        num_classes: int = None

    def forward(self, x):
        return x[:, :self.params.num_classes]

class ActivityOptimizationLayer(AbstractModel, ConsistencyOptimizationMixin):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        consistency_optimization_params: ConsistencyOptimizationParams = field(factory=ConsistencyOptimizationParams)

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self.load_consistency_opt_params()
        self.lat_layer = nn.Linear(self.params.common_params.num_units, self.params.common_params.num_units, bias=self.params.common_params.bias)
        self.back_layer = nn.Linear(self.params.common_params.num_units, self.params.common_params.input_size, bias=self.params.common_params.bias)
        self.activation = self.params.common_params.activation
    
    def forward(self, z, x):
        sh = [z.reshape(-1, z.shape[-1]).T]
        self._optimize_activations(sh, x.reshape(-1, x.shape[-1]).T, 
                                    self.lat_layer.weight, self.lat_layer.bias.reshape(-1,1), 
                                    self.back_layer.weight, self.back_layer.bias.reshape(-1,1), 
                                    activation=self.activation)
        z = self.activation(sh[-1]).T.reshape(*(z.shape))
        return z

class ConsistentActivationMixerBlock(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        channel_mlp_params: CommonModelParams = field(factory=CommonModelParams)
        spatial_mlp_params: CommonModelParams = field(factory=CommonModelParams)
        consistency_optimization_params: ConsistencyOptimizationParams = field(factory=ConsistencyOptimizationParams)
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self._make_network()

    def _make_network(self):
        num_patches = self.params.common_params.input_size[0]
        hidden_size = self.params.common_params.input_size[1]

        mlps_hidden = self.params.spatial_mlp_params.common_params.num_units
        mlpc_hidden = self.params.channel_mlp_params.common_params.num_units

        self.mlps_l1 = nn.Linear(num_patches, mlps_hidden,
                                    bias=self.params.spatial_mlp_params.common_params.bias
                                )
        # self.mlps_activation = self.params.spatial_mlp_params.common_params.activation()
        self.mlps_l2 = nn.Linear(mlps_hidden, num_patches,
                                    bias=self.params.spatial_mlp_params.common_params.bias
                                )
        self.mlpc_l1 = nn.Linear(hidden_size, mlpc_hidden,
                                    bias=self.params.channel_mlp_params.common_params.bias
                                )
        # self.mlpc_activation = self.params.channel_mlp_params.common_params.activation()
        self.mlpc_l2 = nn.Linear(mlpc_hidden,
                                    hidden_size,
                                    bias=self.params.channel_mlp_params.common_params.bias
                                )

        self.mlps_l1_act_opt_layer = ActivityOptimizationLayer(ActivityOptimizationLayer.ModelParams(None,
                                        CommonModelParams(num_patches, mlps_hidden, self.params.spatial_mlp_params.common_params.activation()),
                                        deepcopy(self.params.consistency_optimization_params)
                                    ))
        mlps_l2_act = nn.Sequential(Rearrange('p (b d) -> p b d', d=hidden_size), nn.LayerNorm(hidden_size), Rearrange('p b d -> p (b d)'))
        self.mlps_l2_act_opt_layer = ActivityOptimizationLayer(ActivityOptimizationLayer.ModelParams(None,
                                        CommonModelParams(mlps_hidden, num_patches, mlps_l2_act),
                                        deepcopy(self.params.consistency_optimization_params)
                                    ))
        self.mlpc_l1_act_opt_layer = ActivityOptimizationLayer(ActivityOptimizationLayer.ModelParams(None,
                                        CommonModelParams(hidden_size, mlpc_hidden, self.params.channel_mlp_params.common_params.activation()),
                                        deepcopy(self.params.consistency_optimization_params)
                                    ))
        mlpc_l2_act = nn.Sequential(Rearrange('d n -> n d'), nn.LayerNorm(hidden_size), Rearrange('n d -> d n'))
        self.mlpc_l2_act_opt_layer = ActivityOptimizationLayer(ActivityOptimizationLayer.ModelParams(None,
                                        CommonModelParams(mlpc_hidden, hidden_size, mlpc_l2_act),
                                        deepcopy(self.params.consistency_optimization_params)
                                    ))
        
        # self.mlps_l1_lat = nn.Linear(mlps_hidden, mlps_hidden)
        # self.mlps_l2_back = nn.Linear(mlps_hidden, num_patches)
        # self.mlps_l2_lat = nn.Linear(num_patches, num_patches)
        # self.mlps_l2_back = nn.Linear(num_patches, mlps_hidden)

        # self.mlpc_l1_lat = nn.Linear(mlpc_hidden, mlpc_hidden)
        # self.mlpc_l2_back = nn.Linear(mlpc_hidden, hidden_size)
        # self.mlpc_l2_lat = nn.Linear(hidden_size, hidden_size)
        # self.mlpc_l2_back = nn.Linear(hidden_size, mlpc_hidden)

        self.activation = nn.Identity()
    
    def _get_optimized_activations(self, z, x, lat_layer, back_layer, activation):
        sh = [z.reshape(-1, z.shape[-1]).T]
        self._optimize_activations(sh, x.reshape(-1, x.shape[-1]).T, 
                                    lat_layer.weight, lat_layer.bias.reshape(-1,1), 
                                    back_layer.weight, back_layer.bias.reshape(-1,1), 
                                    activation=activation)
        z = sh[-1].T.reshape(*(z.shape))
        return z

    def _get_spatial_mlp_output(self, x):
        x = x.transpose(1,2)
        
        z1 = self.mlps_l1(x)
        z1 = self.mlps_l1_act_opt_layer(z1, x)
        z2 = self.mlps_l2(z1)
        res = x + z2
        z2 = self.mlps_l2_act_opt_layer(res, z1)

        z2 = z2.transpose(1,2)
        res = res.transpose(1,2)
        return z2, res
    
    def _get_channel_mlp_output(self, x, res):
        z1 = self.mlpc_l1(x)
        z1 = self.mlpc_l1_act_opt_layer(z1, x)
        z2 = self.mlpc_l2(z1)
        res = res + z2
        z2 = self.mlpc_l2_act_opt_layer(res, z1)
        return z2

    def forward(self, x: torch.Tensor):
        z, res = self._get_spatial_mlp_output(x)
        z = self._get_channel_mlp_output(z, res)
        return z


        z: torch.Tensor = x
        z = z.transpose(1,2)
        z = self.spatial_mlp(z)
        z = z.transpose(1,2)
        x_ = x + z
        z = self.layernorm2(x_)
        z = self._get_optimized_activations(z.transpose(1,2), x.transpose(1,2), self.lat1, self.back1).transpose(1,2)
        z = x_ + self.channel_mlp(z)
        z = self.layernorm1(z)
        z = self._get_optimized_activations(z, x, self.lat2, self.back2)
        return z