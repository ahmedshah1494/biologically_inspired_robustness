import torch
from torch import nn
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from attrs import define

class BioNorm(nn.Module):
    def __init__(
            self,
            num_features:int,
            kernel_size:int,
            affine:bool=True,
            device:torch.device=None,
            dtype:torch.dtype=None
        ) -> None:
        r'''
            Applies a biologically motivated local normalization operation to a 4-D input tensor 
            (a mini-batch of 2D inputs with additional channel dimension). This operation exponentiates
            and divides the value at each spatial coordinate by the sum of (exponentiated) neighboring
            values. The neighborhood is defined by the kernel_size.
                R[:, c] = \gamma\frac{X[:, c]^n}{\sigma^n + K^2 AvgPool2d(X, kernel_size=K, stride=1)} + \beta
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

        self.sigma = nn.parameter.Parameter(torch.empty(num_features, **factory_kwargs), requires_grad=False)
        self.pow = nn.parameter.Parameter(torch.empty(num_features, **factory_kwargs), requires_grad=False)
        self.sum_kernel = nn.parameter.Parameter(torch.empty(self.num_features, 1, self.kernel_size, self.kernel_size), requires_grad=False)
        if self.affine:
            self.weight = nn.parameter.Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = nn.parameter.Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()
    
    def extra_repr(self):
        return (
            "{num_features}, kernel_size={kernel_size}".format(**self.__dict__)
        )

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        nn.init.constant_(self.pow, 2)
        nn.init.constant_(self.sigma, 1)
        nn.init.constant_(self.sum_kernel, 1/(self.kernel_size**2))

    def forward(self, x):
        s = self.sigma.view(1,-1,1,1)
        p = self.pow.view(1,-1,1,1)
        w = self.weight.view(1,-1,1,1)
        b = self.bias.view(1,-1,1,1)
        xp = x ** p
        # print(x.min(), x.max())
        suppression_field = nn.functional.conv2d(xp, self.sum_kernel, groups=self.num_features)
        # print(suppression_field.min(), suppression_field.max())
        hks = self.kernel_size // 2
        suppression_field = nn.functional.pad(suppression_field, (hks, hks, hks, hks), mode='replicate')
        # print(suppression_field.min(), suppression_field.max())
        den = s**p + suppression_field
        # print(den.min(), den.max())
        num = xp
        # print(num.min(), num.max())
        xnorm = w * num / den + b
        # print(xnorm.min(), xnorm.max())
        return xnorm

class BioNormWrapper(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        wrapped_class_params: BaseParameters = None
        kernel_size:int = None
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.wrapped_class = params.wrapped_class_params.cls(params.wrapped_class_params)
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
                        module[int(name)] = BioNorm(module[int(name)].num_features, self.params.kernel_size)
                    except:
                        module = setattr(module, name, BioNorm(getattr(module,name).num_features, self.params.kernel_size))

    def forward(self, *args, **kwargs):
        return self.wrapped_class(*args, **kwargs)

    def compute_loss(self, *args, **kwargs):
        return self.wrapped_class.compute_loss(*args, **kwargs)