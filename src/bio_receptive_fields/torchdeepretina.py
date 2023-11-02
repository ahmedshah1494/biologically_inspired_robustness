import torch
from torch.nn import ReLU, Tanh
from attrs import define
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from torch import nn
from torchdeepretina import models
from torchdeepretina.custom_modules import (AbsBatchNorm1d, AbsBatchNorm2d, Exponential,
                                            Flatten, GrabUnits,
                                            LinearStackedConv2d,
                                            OneToOneLinearStackedConv2d,
                                            Reshape, SelfAttn2d, update_shape)

class GaussianNoise(nn.Module):
    def __init__(self, std=0.1, trainable=False, adapt=False,
                                               momentum=.95,
                                               deterministic=False):
        """
        std - float
            the standard deviation of the noise to add to the layer.
            if adapt is true, this is used as the proportional value to
            set the std to based of the std of the activations.
            gauss_std = activ_std*std
        trainable - bool
            If trainable is set to True, then the std is turned into
            a learned parameter. Cannot be set to true if adapt is True
        adapt - bool
            adapts the gaussian std to a proportion of the
            std of the received activations. Cannot be set to True if
            trainable is True
        momentum - float (0 <= momentum < 1)
            this is the exponentially moving average factor for
            updating the activ_std. 0 uses the std of the current
            activations.
        """
        super(GaussianNoise, self).__init__()
        self.trainable = trainable
        self.adapt = adapt
        assert not (self.trainable and self.adapt)
        self.std = std
        self.sigma = nn.Parameter(torch.ones(1)*std,
                            requires_grad=trainable)
        self.running_std = 1
        self.deterministic = deterministic
        self.momentum = momentum if adapt else None

    def forward(self, x):
        if self.std == 0:
            return x
        if self.adapt:
            xstd = x.std().item()
            self.running_std = self.momentum*self.running_std +\
                                          (1-self.momentum)*xstd
            self.sigma.data[0] = self.std*self.running_std
        if self.deterministic:
            noise = self.sigma*torch.empty_like(x).uniform_(generator=torch.Generator(device=x.device).manual_seed(69868695))
        else:
            noise = self.sigma*torch.randn_like(x)
        return x + noise

    def extra_repr(self):
        s = 'std={}, trainable={}, adapt={}, momentum={}'
        return s.format(self.std, self.trainable,
                        self.adapt, self.momentum)

class VaryModel(models.TDRModel):
    """
    Built as a modular model that can assume the form of most other
    models in this package.

    drop_p: float
            the dropout probability for the linearly stacked
            convolutions
        one2one: bool
            if true and stackconvs is true, then the stacked
            convolutions do not allow crosstalk between the inner
            channels
        stack_ksizes: list of ints
            the kernel size of the stacked convolutions
        stack_chans: list of ints
            the channel size of the stacked convolutions. If none,
            defaults to channel size of main convolution
        paddings: list of ints
            the padding for each conv layer. If none,
            defaults to 0.
        self_attn: bool
            if true, a SelfAttn2d module is added to layers 1 and 2.
            (See `custom_modules.SelfAttn2d` for details)
    """
    def __init__(self, drop_p=0, one2one=False, stack_ksizes=3,
                                        stack_chans=None,
                                        paddings=None,
                                        self_attn=False,
                                        **kwargs):
        super().__init__(**kwargs)
        self.name = 'VaryNet'
        self.drop_p = drop_p
        self.one2one = one2one
        self.self_attn = self_attn
        if isinstance(stack_ksizes, int):
            stack_ksizes=[stack_ksizes for i in\
                                        range(len(self.ksizes))]
        self.stack_ksizes = stack_ksizes
        if stack_chans is None or isinstance(stack_chans, int):
            stack_chans = [stack_chans for i in\
                                        range(len(self.ksizes))]
        self.stack_chans = stack_chans
        self.paddings = [0 for x in stack_ksizes] if paddings is None\
                                                         else paddings
        shape = self.img_shape[1:] # (H, W)
        self.shapes = []
        modules = []

        #### Layer Loop
        temp_chans = [self.img_shape[0]]
        if self.n_layers > 1:
            temp_chans = [self.img_shape[0]]+self.chans
        for i in range(self.n_layers-1):
            ## Convolution
            if not self.stackconvs:
                conv = nn.Conv2d(temp_chans[i],temp_chans[i+1],
                                    kernel_size=self.ksizes[i],
                                    padding=self.paddings[i],
                                    bias=self.bias,
                                    groups=self.groups if i>0 else 1)
            else:
                if self.one2one:
                    conv = OneToOneLinearStackedConv2d(temp_chans[i],
                                          temp_chans[i+1],
                                          kernel_size=self.ksizes[i],
                                          padding=self.paddings[i],
                                          bias=self.bias)
                else:
                    conv = LinearStackedConv2d(temp_chans[i],
                                       temp_chans[i+1],
                                       kernel_size=self.ksizes[i],
                                       abs_bnorm=False,
                                       bias=self.bias,
                                       stack_chan=stack_chans[i],
                                       stack_ksize=stack_ksizes[i],
                                       drop_p=self.drop_p,
                                       padding=self.paddings[i],
                                       groups=self.groups if i>0 else 1)
            modules.append(conv)
            shape = update_shape(shape, self.ksizes[i],
                                        padding=self.paddings[i])
            self.shapes.append(tuple(shape))
            if self.self_attn and i>0: # first layer doesn't get attn
                attn = SelfAttn2d(temp_chans[i+1], self.attn_size,
                                                   self.n_heads,
                                                   self.prob_attn)
                modules.append(attn)
                    
            if self.bnaftrelu:
                modules.append(GaussianNoise(std=self.noise))
                modules.append(globals()[self.activ_fxn]())
            ## BatchNorm
            if self.bnorm_d == 1:
                modules.append(Flatten())
                size = temp_chans[i+1]*shape[0]*shape[1]
                modules.append(AbsBatchNorm1d(size,eps=1e-3,
                             momentum=self.bn_moment))
                modules.append(Reshape((-1,temp_chans[i+1],*shape)))
            else:
                bnorm = AbsBatchNorm2d(temp_chans[i+1],eps=1e-3,
                                        momentum=self.bn_moment)
                modules.append(bnorm)
            # Noise and Activation
            if not self.bnaftrelu:
                modules.append(GaussianNoise(std=self.noise))
                modules.append(globals()[self.activ_fxn]())

        ##### Final Layer
        if self.convgc:
            ksize = self.ksizes[self.n_layers-1]
            if self.finalstack:
                conv_type = LinearStackedConv2d
            else:
                conv_type = nn.Conv2d
            conv = conv_type(temp_chans[-1], self.n_units,
                                            kernel_size=ksize,
                                            bias=self.gc_bias)
            modules.append(conv)
            shape = update_shape(shape, self.ksizes[self.n_layers-1])
            self.shapes.append(tuple(shape))
            if self.self_attn:
                attn = SelfAttn2d(self.n_units, self.attn_size,
                                                self.n_heads,
                                                self.prob_attn)
                modules.append(attn)
            modules.append(GrabUnits(self.centers,
                                     self.ksizes[:self.n_layers],
                                     self.img_shape))
        else:
            modules.append(Flatten())
            modules.append(nn.Linear(temp_chans[-1]*shape[0]*shape[1],
                                                 self.n_units,
                                                 bias=self.gc_bias))
        modules.append(AbsBatchNorm1d(self.n_units, eps=1e-3,
                                    momentum=self.bn_moment))
        if self.softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))

        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

    def deactivate_grads(self, deactiv=True):
        """
        Turns grad off for all trainable parameters in model
        """
        for p in self.parameters():
            p.requires_grad = deactiv

    def tiled_forward(self,x):
        """
        Allows for the fully convolutional functionality
        """
        if not self.convgc:
            return self.forward(x)
        # Remove GrabUnits layer
        fx = self.sequential[:-3](x) 
        bnorm = self.sequential[-2]
        # Perform 2dbatchnorm using 1d parameters
        fx =torch.nn.functional.batch_norm(fx,bnorm.running_mean.data,
                                            bnorm.running_var.data,
                                            weight=bnorm.scale.abs(),
                                            bias=bnorm.shift,
                                            eps=bnorm.eps,
                                            momentum=bnorm.momentum,
                                            training=self.training)
        fx = self.sequential[-1:](fx)
        if not self.training and self.infr_exp:
            return torch.exp(fx)
        return fx

    def determinise_noise(self):
        for module in self.sequential:
            if isinstance(module, GaussianNoise):
                module.deterministic = True
    
    def randomize_noise(self):
        for module in self.sequential:
            if isinstance(module, GaussianNoise):
                module.deterministic = False

class TorchDeepRetina(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        ckp_path: str = None
        final_layer_idx: int = 8
        freeze: bool = True
        upscale_factor: int = 1
    
    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self._make_network()

    def _make_network(self):
        ckp = torch.load(self.params.ckp_path)
        self.retina_ = VaryModel(**ckp)
        self.retina_.load_state_dict(ckp['model_state_dict'])
        self.retina = self.retina_.sequential[:self.params.final_layer_idx]
        if self.params.freeze:
            self.retina.requires_grad_(False)
            self.retina = self.retina.eval()

    def determinise_noise(self):
        for module in self.retina:
            if isinstance(module, GaussianNoise):
                module.deterministic = True
    
    def randomize_noise(self):
        for module in self.retina:
            if isinstance(module, GaussianNoise):
                module.deterministic = False

    def forward(self, x:torch.Tensor):
        if self.training:
            self.randomize_noise()
        else:
            self.determinise_noise()
        b, c, h, w = x.shape
        x = x.reshape(-1, 1, h, w).repeat_interleave(40, 1)
        x = self.retina(x)
        x = x.reshape(b, c, *(x.shape[1:]))
        b, c1, c2, h, w = x.shape
        x = x.reshape(b, c1*c2, h, w)
        if self.params.upscale_factor > 1:
            x = torch.nn.functional.interpolate(x, scale_factor=self.params.upscale_factor)
        return x
    
    def compute_loss(self, x, y, return_logits=True):
        out = self.forward(x)
        logits = out
        loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss