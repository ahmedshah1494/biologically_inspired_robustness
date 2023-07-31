import torch
from torch import nn
from adversarialML.biologically_inspired_models.src.bio_receptive_fields.utils import bivariate_gaussian_kernels
from adversarialML.biologically_inspired_models.src.bio_receptive_fields.dog_filterbank import _DoGFilterbank, DoGFilterbank
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from attrs import define

class _LocalContrast(nn.Module):
    def __init__(self, kernel_size, in_channels, stride=1, eps=1e-2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.stride = stride
        self.eps = eps

        std = kernel_size / 4
        self.kernel = bivariate_gaussian_kernels(
                            kernel_size,
                            torch.FloatTensor([[std, std]]),
                            torch.zeros(1,)
                    ).repeat_interleave(self.in_channels, 0).unsqueeze(1)
        self.kernel = nn.parameter.Parameter(self.kernel, requires_grad=False)
    
    def forward(self, x):
        hks = self.kernel_size // 2
        xpad = nn.functional.pad(x, (hks, hks, hks, hks), mode='replicate')
        mean = nn.functional.conv2d(xpad, self.kernel, stride=self.stride, groups=self.in_channels)
        if self.stride > 1:
            x = nn.functional.interpolate(x, scale_factor=1/self.stride)
        x = (x - mean) / (self.eps+mean)
        if not torch.isfinite(x).all():
            print(x.min(), x.max())
            print(mean.min(), mean.max())
            exit()
        return x

class LocalContrast(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        kernel_size: int = None
        in_channels: int = None
        stride: int = 1
        eps: float = 1e-5
    
    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self._make_network()

    def _make_network(self):
        kwargs_to_exclude = set(['cls'])
        kwargs = self.params.asdict(filter=lambda a,v: a.name not in kwargs_to_exclude)
        self.LC = _LocalContrast(**kwargs)
    
    def forward(self, x):
        return self.LC(x)
    
    def compute_loss(self, x, y, return_logits=True):
        out = self.forward(x)
        logits = out
        loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

class _ContrastNormalize(nn.Module):
    '''
        Perform contrast normalizations. The following steps are performed:
            1 - Local contrast computation using LocalContrast
            2 - Applying DoG filterbank
            3 - Computing mean local contrast by convolving 1 with a Gaussian kernel.
            4 - Dividing the output of 2 by 3
            5 - Scaling and translating 4
    '''
    def __init__(self,
                 local_contrast: _LocalContrast,
                 DoGFB: _DoGFilterbank,
                 kernel_size:int,
                 stride:int=1,
                 eps=1e-5,
                 affine=True
                 ) -> None:
        super().__init__()
        self.local_contrast = local_contrast
        self.DoGFB = DoGFB
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps
        self.affine = affine
        self.num_channels = DoGFB.get_num_kernels()
        self.in_channels = 3

        self._initialize_kernel()
        self._initialize_parameters()

    def _initialize_kernel(self):
        stds = self.DoGFB.get_kernel_stds().max(1)[0] * 1.5
        # stds = torch.stack([stds, stds], 1).repeat_interleave(3, 0)
        # self.kernel = bivariate_gaussian_kernels(
        #                     self.kernel_size,
        #                     stds,
        #                     torch.zeros(stds.shape[0],)
        #             ).unsqueeze(1)
        self.kernel = bivariate_gaussian_kernels(
                            self.kernel_size,
                            torch.zeros(1,2)+stds.max(),
                            torch.zeros(1,)
                    ).unsqueeze(1).repeat_interleave(self.in_channels, 0)
        self.kernel = nn.parameter.Parameter(self.kernel, requires_grad=False)
    
    def _initialize_parameters(self):
        if self.affine:
            self.weight = nn.parameter.Parameter(torch.empty(1, self.num_channels, 1, 1))
            self.bias = nn.parameter.Parameter(torch.empty(1, self.num_channels, 1, 1))
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
    
    def _compute_mean_local_contrast(self, x):
        hks = self.kernel_size // 2
        x = nn.functional.pad(x, (hks, hks, hks, hks), mode='replicate')
        mean = nn.functional.conv2d(x, self.kernel, stride=self.stride, groups=self.in_channels)        
        return mean
    
    def forward(self, x):
        b, c, h, w = x.shape
        lc = self.local_contrast(x)
        dog_lc = self.DoGFB(lc)
        dog_lc = dog_lc.view(-1, 3, dog_lc.shape[2], dog_lc.shape[3])
        sf = dog_lc.shape[-1] / lc.shape[-1]
        if sf != 1:
            lc = nn.functional.interpolate(lc, dog_lc.shape[-2:], mode='bilinear')
        contrast_var = self._compute_mean_local_contrast(lc ** 2)
        contrast_std = torch.sqrt(contrast_var)
        cn = dog_lc / (self.eps + contrast_std.repeat_interleave(self.num_channels // self.in_channels, 0))
        cn = cn.view(b, -1, cn.shape[2], cn.shape[3])
        cn = self.weight * cn + self.bias
        return cn

class ContrastNormalize(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        local_contrast_params: LocalContrast.ModelParams = None
        DoGFB_params: DoGFilterbank.ModelParams = None
        kernel_size: int = None
        stride: int = 1
        eps: float = 1e-5
        affine: bool = True
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()
    
    def _make_network(self):
        lc = self.params.local_contrast_params.cls(self.params.local_contrast_params).LC

        self.params.DoGFB_params.application_mode = 'all_channels_replicated'
        self.params.DoGFB_params.normalize_input = False
        self.params.DoGFB_params.activation = nn.Identity
        dogfb = self.params.DoGFB_params.cls(self.params.DoGFB_params).filterbank

        self.CN = _ContrastNormalize(lc, dogfb, self.params.kernel_size, self.params.stride,
                                     self.params.eps, self.params.affine)
    
    def forward(self, x):
        return self.CN(x)

    def compute_loss(self, x, y, return_logits=True):
        out = self.forward(x)
        logits = out
        loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss