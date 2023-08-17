from typing import Literal, Union, List, Callable
import torch
from torch import nn
import numpy as np
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from attrs import define
from adversarialML.biologically_inspired_models.src.bio_receptive_fields.utils import eliptical_dogkerns

class NormalizationLayer(nn.Module):
    def __init__(self, mean: Union[float, List[float]] = [0.485, 0.456, 0.406],
                 std: Union[float, List[float]] = [0.229, 0.224, 0.225]) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        if isinstance(self.mean, list):
            self.mean = nn.parameter.Parameter(torch.FloatTensor(self.mean).reshape(1,-1,1,1), requires_grad=False)
        if isinstance(self.std, list):
            self.std = nn.parameter.Parameter(torch.FloatTensor(self.std).reshape(1,-1,1,1), requires_grad=False)
    
    def __repr__(self):
        return f'NormalizationLayer(mean={self.mean}, std={self.std})'
    
    def forward(self, x):
        return (x-self.mean)/self.std

class GreyscaleLayer(nn.Module):
    def __init__(self, channel_wt: List[float] = [0.2989, 0.5870, 0.1140]) -> None:
        super().__init__()
        self.channel_wt = nn.parameter.Parameter(torch.FloatTensor(channel_wt).unsqueeze(1,-1,1,1), requires_grad=False)
    
    def __repr__(self):
        return f'GreyscaleLayer(wt={self.channel_wt})'

    def forward(self, x):
        return (x * self.channel_wt).sum(1, keepdim=True)

class RandomFilterbank(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        kernel_size:int=None
        in_channels:int = None
        out_channels:int = None
        stride:int = 1
        padding:int = 0
        dilation:int = 1
        activation: Callable = nn.ReLU
        normalize_input: bool = True
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        kwargs_to_exclude = set(['cls', 'activation', 'normalize_input'])
        kwargs = self.params.asdict(filter=lambda a,v: a.name not in kwargs_to_exclude)
        self.conv = nn.Conv2d(**kwargs).requires_grad_(False)
        self.preprocess = NormalizationLayer() if self.params.normalize_input else nn.Identity()
        self.activation = params.activation()
    
    def forward(self, x):
        x = self.preprocess(x)
        x = self.conv(x)
        x = self.activation(x)
        return x
    
    def compute_loss(self, x, y, return_logits=True):
        out = self.forward(x)
        logits = out
        loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

class _DoGFilterbank(nn.Module):
    def __init__(self, 
                 kernel_size:int,
                 in_channels:int = 3,
                 stride:int = 1,
                 padding:int = 0,
                 dilation:int = 1,
                 activation: Callable = nn.ReLU,
                 normalize_input: bool = True,
                 num_scales:int=8,
                 num_shapes:int=8,
                 num_orientations:int=8,
                 base_std:float=1.6,
                 scaling_factor:float=2,
                 ratio:float=np.sqrt(2),
                 min_std:float=0.4, 
                 max_corr:float=.9,
                 application_mode:Literal['greyscale', 
                                    'random_channel',
                                    'all_channels',
                                    'single_opponent',
                                    'all_channels_replicated']='random_channel',
                 kernel_path:str=None) -> None:
        '''
        Creates a filterbank of oriented Difference (or Laplacian) of Gaussian filters. The DoG filter
        is computed as $$\phi(M;0,\sigma_x,\sigma_y, \\rho)-\phi(M;0,ratio*\sigma_x, ratio*\sigma_y, \\rho)$$,
        where $$M$$ is a square grid over [-K/2, K/2], $$\phi$$ is the bivariate Gaussian PDF, $$\sigma_x$$ and
        $$\sigma_y$$ are the standard deviation values of the horizontal and vertical dimensions, and $$\\rho$$ 
        is their correlation. $$\sigma_x$$ and $$\sigma_y$$ control the height and width of the receptive field,
        while $$\\rho$$ controls its orientation.
        
        DoG filters are computed for num_scales different scales. At scale, s, we compute 2 x num_shapes x num_orientations
        DoG kernels. To get different shapes we set the std along one dim being scaling_factor**s*base_std, and set the
        std at the other dimension to values in linspace(min_std, scaling_factor**s*base_std, num_shapes). We repeat 
        the latter for both the vertical and horizontal dimension. To get different orientations we set $$\\rho$$ in 
        linspace(-1, 1, num_orientations). 
        
        Therefore the total number of filters are N = num_scales x 2 x num_shapes x num_orientations.

        Inputs:
            - kernel_size (int):    The length of a square kernel
            - num_scales (int):     Specifies the number of different lengths of the major axis of eliptical DoG filters to be 
                                    included in the filterbank.
            - num_shapes (int):     Specifies the number of different lengths of the minor axis of eliptical DoG filters to be 
                                    included in the filterbank.
            - num_orientations (int): Specifies the number of orientations of the filters in the bank.
            - base_std (float):     The std corresponding to the lowest scale. It is proportional to the smallest length of the
                                    major axis.
            - ratio (float):        The ratio of the std of the negative to the std of the positive Gaussian.
            - min_std (float):      The std corresponding to the thinnest elipse. Proportional to the the length of the smallest
                                    minor axis.
            - application_mode (Literal['greyscale','random_channel','all_channels','single_opponent']):    Defines the channels
                                    to which the kernel is applied.
                                    - greyscale: The kernels are applied to a greyscaled version of the input
                                    - random_channel: Each kernel is applied to one randomly chosen input channel
                                    - all_channels: Each kernel is applied to all input channels with the same weights
                                    - single_opponent: The excitatory (+ve) and inhibitory (-ve) components of the kernel are
                                                       applied to different channels. The filter is replicated to cover all pairs
                                                       of input channels. Note that this mode increases the output filters by a
                                                       factor of (in_channels choose 2).
        '''
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.activation = activation()
        self.normalize_input = normalize_input
        self.num_scales = num_scales
        self.num_shapes = num_shapes
        self.num_orientations = num_orientations
        self.base_std = base_std
        self.scaling_factor = scaling_factor
        self.ratio = ratio
        self.min_std = min_std
        self.max_corr = max_corr
        self.application_mode = application_mode
        self.kernel_path = kernel_path

        self._initialize()

    def _initialize(self):
        self.preprocess = NormalizationLayer() if self.normalize_input else nn.Identity()
        if self.kernel_path is not None:
            self.kernels = torch.load(self.kernel_path, map_location=torch.device('cpu'))
            self.kernels = nn.parameter.Parameter(self.kernels, requires_grad=False)
            return

        base_stds = torch.stack(
            [torch.FloatTensor([self.base_std]*self.num_shapes),
            torch.linspace(self.min_std, self.base_std, self.num_shapes)],
            1)
        base_stds = torch.cat([base_stds, torch.flip(base_stds, [1])], 0)

        stds = torch.cat([(self.scaling_factor**s)*base_stds for s in range(self.num_scales)], 0)
        corrs = torch.linspace(-self.max_corr, self.max_corr, self.num_orientations)

        stdsi_x_corrsi = torch.cartesian_prod(torch.arange(stds.shape[0]), torch.arange(corrs.shape[0]))
        self.stds = stds[stdsi_x_corrsi[:,0]]
        self.corrs = corrs[stdsi_x_corrsi[:,1]]

        self.kernels = eliptical_dogkerns(self.kernel_size, self.stds, self.corrs, ratio=self.ratio).unsqueeze(1)
        self.channel_weights = nn.parameter.Parameter(torch.ones(1, self.in_channels, 1, 1), requires_grad=False)        
        self.kernels = nn.parameter.Parameter(self.kernels, requires_grad=False)
        self._setup_application_mode()
    
    def _setup_application_mode(self):
        if self.application_mode == 'greyscale':
            self.preprocess = nn.Sequential(
                GreyscaleLayer(),
                NormalizationLayer([0.5]*3, [0.5]*3)
            )
        else:
            if self.application_mode == 'random_channel':
                # active_channels = torch.randint(0, self.in_channels, (self.kernels.shape[0],))
                active_channels = torch.multinomial(torch.FloatTensor([0.64, 0.32, 0.02]), self.kernels.shape[0], replacement=True)
                new_kernels = torch.zeros_like(self.kernels).repeat_interleave(self.in_channels, 1)
                new_kernels[torch.arange(0, new_kernels.shape[0]), active_channels] = self.kernels[:,0]
            if self.application_mode == 'all_channels':
                new_kernels = self.kernels.repeat_interleave(self.in_channels, 1)
            if self.application_mode == 'all_channels_learnable':
                new_kernels = self.kernels.repeat_interleave(self.in_channels, 1)
                nn.init.uniform_(self.channel_weights.data, -np.sqrt(6/self.in_channels), -np.sqrt(6/self.in_channels))
                self.channel_weights.requires_grad_(True)
            if self.application_mode == 'all_channels_replicated':
                new_kernels = torch.zeros(self.kernels.shape[0], self.in_channels, self.in_channels, self.kernels.shape[-2], self.kernels.shape[-1])
                for j in range(self.in_channels):
                    new_kernels[:, j, [j]] = self.kernels
                new_kernels = torch.flatten(new_kernels, 0, 1)
            if self.application_mode == 'single_opponent':
                from itertools import combinations            
                _excitatory = torch.relu(self.kernels)
                _inhibitory = torch.relu(-self.kernels)
                channel_combinations = list(combinations(range(self.in_channels), 2))
                new_kernels = torch.zeros(self.kernels.shape[0], len(channel_combinations), self.in_channels, self.kernels.shape[-2], self.kernels.shape[-1])
                for j, (exC, inC) in enumerate(channel_combinations):
                    new_kernels[:, j, [exC]] = _excitatory
                    new_kernels[:, j, [inC]] = -_inhibitory
                new_kernels = torch.flatten(new_kernels, 0, 1)
            self.kernels.data = new_kernels
    
    def get_num_kernels(self):
        return self.kernels.shape[0]

    def get_kernel_stds(self):
        return self.stds

    def forward(self, x):
        x = self.preprocess(x)
        K = self.kernels * self.channel_weights
        x = nn.functional.conv2d(x, K, stride=self.stride, padding=self.padding, dilation=self.dilation)
        x = self.activation(x)
        return x

class DoGFilterbank(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        kernel_size:int=None
        in_channels:int = 3
        stride:int = 1
        padding:int = 0
        dilation:int = 1
        activation: Callable = nn.ReLU
        normalize_input: bool = True
        num_scales:int=8
        num_shapes:int=8
        num_orientations:int=8
        base_std:float=1.6
        scaling_factor:float=2
        ratio:float=np.sqrt(2)
        min_std:float=0.1
        max_corr:float=.9
        application_mode:Literal['greyscale', 
                        'random_channel',
                        'all_channels',
                        'single_opponent']='random_channel'
        kernel_path:str=None

    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self._make_network()
    
    def _make_network(self):
        kwargs_to_exclude = set(['cls'])
        kwargs = self.params.asdict(filter=lambda a,v: a.name not in kwargs_to_exclude)
        self.filterbank = _DoGFilterbank(**kwargs)
    
    def forward(self, x):
        return self.filterbank(x)

    def compute_loss(self, x, y, return_logits=True):
        out = self.forward(x)
        logits = out
        loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

class _LearnableDoGFilterbank(nn.Module):
    def __init__(self,
                 num_filters:int,
                 kernel_size:int,
                 in_channels:int = 3,
                 stride:int = 1,
                 padding:int = 0,
                 dilation:int = 1,
                 activation: Callable = nn.ReLU,
                 normalize_input: bool = True,
                 ratio:float=np.sqrt(2),
                 min_std:float=0.4, 
                 max_corr:float=.9) -> None:
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.activation = activation()
        self.normalize_input = normalize_input
        self.ratio = ratio
        self.min_std = min_std
        self.max_std = self.kernel_size / 4
        self.max_corr = max_corr

        self.stds_raw = nn.parameter.Parameter(nn.init.kaiming_uniform_(torch.empty(self.num_filters*self.in_channels, 2), mode='fan_out'), requires_grad=True)
        self.corrs_raw = nn.parameter.Parameter(nn.init.uniform_(torch.empty(self.num_filters*self.in_channels), -np.sqrt(6/self.num_filters), np.sqrt(6/self.num_filters)), requires_grad=True)
        self.preprocess = NormalizationLayer() if self.normalize_input else nn.Identity()

        self._initialize_kernel_grid()
    
    def _initialize_kernel_grid(self):
        x = torch.arange(-(self.kernel_size//2), self.kernel_size//2+1, 1)
        grid = torch.stack(torch.meshgrid(x, x), -1) # M x M x 2
        grid = grid.unsqueeze(0).repeat_interleave(self.num_filters * self.in_channels, 0).to(self.stds_raw.device) # N x M x M x 2        
        self.grid2 = nn.parameter.Parameter(grid**2, requires_grad=False)
        self.xy = nn.parameter.Parameter(torch.prod(grid, -1), requires_grad=False)

    def _generate_filterbank(self):
        stds = (torch.sigmoid(self.stds_raw) * 0.6 + 0.4) * self.max_std
        corrs = torch.tanh(self.corrs_raw) * self.max_corr

        N = self.num_filters * self.in_channels
        stds = stds.view(N, 1, 1, 2)
        stds2 = stds**2
        stds_prod = torch.prod(stds, -1)

        corrs = corrs.view(N, 1, 1)
        corrs2 = corrs ** 2

        kernels = torch.exp(- ((self.grid2 / stds2).sum(-1) - 2*(self.xy/stds_prod)*corrs) / (2*(1-corrs2+1e-8)))
        kernels = kernels / (2*np.pi*stds_prod*torch.sqrt(1 - corrs2)+1e-8)

        # kernels = eliptical_dogkerns(self.kernel_size, stds, corrs, ratio=self.ratio).unsqueeze(1)
        kernels = kernels.reshape(self.num_filters, self.in_channels, self.kernel_size, self.kernel_size)
        return kernels
    
    def forward(self, x):
        kernels = self._generate_filterbank()
        x = self.preprocess(x)
        x = nn.functional.conv2d(x, kernels, stride=self.stride, padding=self.padding, dilation=self.dilation)
        x = self.activation(x)
        return x

class LearnableDoGFilterbank(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        num_filters:int = None
        kernel_size:int = None
        in_channels:int = 3
        stride:int = 1
        padding:int = 0
        dilation:int = 1
        activation: Callable = nn.ReLU
        normalize_input: bool = True
        ratio:float=np.sqrt(2)
        min_std:float=0.4,
        max_corr:float=.9

    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self._make_network()
    
    def _make_network(self):
        kwargs_to_exclude = set(['cls'])
        kwargs = self.params.asdict(filter=lambda a,v: a.name not in kwargs_to_exclude)
        self.filterbank = _LearnableDoGFilterbank(**kwargs)
    
    def forward(self, x):
        return self.filterbank(x)

    def compute_loss(self, x, y, return_logits=True):
        out = self.forward(x)
        logits = out
        loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss