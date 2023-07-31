import torch
from torch import nn
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from attrs import define

class _BioNorm(nn.Module):
    def __init__(
            self,
            num_features:int,
            kernel_size:int,
            affine:bool=True,
            m: int = 2,
            n: int = 1,
            p: int = 1/2,
            sigma: int = 1e-2,
            center: bool = False,
            device:torch.device=None,
            dtype:torch.dtype=None
        ) -> None:
        r'''
            Applies a biologically motivated local normalization operation to a 4-D input tensor 
            (a mini-batch of 2D inputs with additional channel dimension). This operation exponentiates
            and divides the value at each spatial coordinate by the sum of (exponentiated) neighboring
            values. The neighborhood is defined by the kernel_size.
                R[:, c] = \gamma\frac{X[:, c]^n}{\sigma^n + (K^2 AvgPool2d(X, kernel_size=K, stride=1))^\frac{n}{p}} + \beta
            where \sigma, n, \gamma and \beta are trainable parameters.
            Args:
                - num_features (int): number of channels in the input
                - kernel_size (int): the size of the kernel used to compute the normalizing factor
                - affine (bool): If ``False`` \gamma and \beta are removed. Default=``True``
        '''
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.affine = affine
        self.n = n
        self.m = m
        self.p = p
        self.center = center

        self.sigma = sigma
        self.sum_kernel = nn.parameter.Parameter(torch.empty(1, self.num_features, self.kernel_size, self.kernel_size), requires_grad=False)
        if self.affine:
            self.weight = nn.parameter.Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = nn.parameter.Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()
    
    def extra_repr(self):
        return (
            "num_features={num_features}, kernel_size={kernel_size}, affine={affine}, centered={center}".format(**self.__dict__)
        )

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        nn.init.constant_(self.sum_kernel, 1/(self.num_features*self.kernel_size**2))

    def forward(self, x):
        hks = self.kernel_size // 2
        # print(x.min(), x.max())
        if self.center:
            mean = nn.functional.conv2d(x, self.sum_kernel)
            mean = nn.functional.pad(mean, (hks, hks, hks, hks), mode='replicate')
            # print(mean.min(), mean.max())
            x = x - mean
        # print(x.min(), x.max())
        num = x ** self.n
        # print(num.min(), num.max())

        var = nn.functional.conv2d(x ** self.m, self.sum_kernel)
        var = nn.functional.pad(var, (hks, hks, hks, hks), mode='replicate')
        # print(var.min(), var.max())
        den = (self.sigma + var)**self.p
        # print(den.min(), den.max())
        xnorm = num / den
        if self.affine:
            w = self.weight.view(1,-1,1,1)
            b = self.bias.view(1,-1,1,1)
            xnorm = w * xnorm + b
        # print(xnorm.min(), xnorm.max())
        if (not torch.isfinite(xnorm).all()):
            print(xnorm.min(), xnorm.max())
            print(num.min(), num.max())
            print(den.min(), den.max())
            exit()
        return xnorm

class BioNorm(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        num_features:int = None
        kernel_size:int = None
        affine:bool=True
        m: int = 2
        n: int = 1
        p: int = 1/2
        sigma: int = 1e-2
        center: bool = False
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self._make_network()
    
    def _make_network(self):
        kwargs_to_exclude = set(['cls'])
        kwargs = self.params.asdict(filter=lambda a,v: a.name not in kwargs_to_exclude)
        self.bn = _BioNorm(**kwargs)
    
    def forward(self, x):
        return self.bn(x)

    def compute_loss(self, x, y, return_logits=True):
        out = self.forward(x)
        logits = out
        loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

class BioNormWrapper(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        wrapped_class_params: BaseParameters = None
        bionorm_params: BioNorm.ModelParams = None
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.wrapped_class = params.wrapped_class_params.cls(params.wrapped_class_params)
        kwargs_to_exclude = set(['cls', 'num_features'])
        self.bionorm_kwargs = params.bionorm_params.asdict(filter=lambda a,v: a.name not in kwargs_to_exclude)
        self._replace_batchnorm_with_bionorm()
    
    def _replace_batchnorm_with_bionorm(self):
        batch_norm_paths = [n.split('.') for n,m in self.wrapped_class.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
        for pth in batch_norm_paths:
            module = self.wrapped_class
            for j, name in enumerate(pth):
                if j < len(pth)-1:
                    try:
                        module = module[int(name)]
                    except:
                        module = getattr(module, name)
                else:
                    try:
                        module[int(name)] = _BioNorm(module[int(name)].num_features, **(self.bionorm_kwargs))
                    except:
                        module = setattr(module, name, _BioNorm(getattr(module,name).num_features, **(self.bionorm_kwargs)))


    def forward(self, *args, **kwargs):
        return self.wrapped_class(*args, **kwargs)

    def compute_loss(self, *args, **kwargs):
        return self.wrapped_class.compute_loss(*args, **kwargs)

# class ConvBioNorm(nn.Module):
#     def __init__(self, 
#                  in_channels:int,
#                  out_channels:int,
#                  kernel_size_1:int,
#                  kernel_size_2:int,
#                  stride:int = 1,
#                  padding:int = 0,
#                  dilation:int = 1,
#                  bias:bool = True,
#                  n:int = 1,
#                  m:int = 1,
#                  p:int = 1):
#         super().__init__()
#         self.kernel_size_1 = kernel_size_1
#         self.kernel_size_2 = kernel_size_2
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.n = n
#         self.m = m
#         self.p = p

#         self.drive = nn.Conv2d(self.in_channels,
#                                 self.out_channels,
#                                 self.kernel_size_1,
#                                 stride=self.stride,
#                                 padding=self.padding,
#                                 dilation=self.dilation,
#                                 bias=self.bias)
#         pad_offset = (self.kernel_size_2 - self.kernel_size_1) // 2
#         self.norm = nn.Conv2d(self.in_channels,
#                                 self.out_channels,
#                                 self.kernel_size_2,
#                                 stride=self.stride,
#                                 padding=self.padding+pad_offset,
#                                 dilation=self.dilation,
#                                 bias=self.bias)
#         nn.init.constant_(self.norm.bias, 1e-5)
    
#     def forward(self, x):
#         drive = self.drive(x ** self.n)
#         sup = self.norm(x ** self.m) ** self.p
#         out = drive / sup
#         return out
    
# class ConvBioNormWrapper(AbstractModel):
#     @define(slots=False)
#     class ModelParams(BaseParameters):
#         wrapped_class_params: BaseParameters = None
#         n:int = 1
#         m:int = 1
#         p:int = 1
    
#     def __init__(self, params: ModelParams) -> None:
#         super().__init__(params)
#         self.wrapped_class = params.wrapped_class_params.cls(params.wrapped_class_params)
#         self._replace_batchnorm_with_bionorm()
    
#     def _replace_conv_with_conv_bionorm(self):
#         batch_norm_paths = [n.split('.') for n,m in self.wrapped_class.named_modules() if isinstance(m, torch.nn.Conv2d)]
#         for pth in batch_norm_paths:
#             module = self.wrapped_class
#             for j, name in enumerate(pth):
#                 if j < len(pth)-1:
#                     try:
#                         module = module[int(name)]
#                     except:
#                         module = getattr(module, name)
#                 else:
#                     try:
#                         conv = module[int(name)]
#                         module[int(name)] = ConvBioNorm(
#                             conv.in_channels,
#                             conv.out_channels,
#                             conv.kernel_size,
#                             ,
#                             stride=conv.stride,
#                             padding=conv.padding
#                         )
#                     except:
#                         module = setattr(module, name, BioNorm(getattr(module,name).num_features, self.params.kernel_size))

#     def forward(self, *args, **kwargs):
#         return self.wrapped_class(*args, **kwargs)

#     def compute_loss(self, *args, **kwargs):
#         return self.wrapped_class.compute_loss(*args, **kwargs)