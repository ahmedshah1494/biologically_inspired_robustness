import time
from typing import Callable, List, Tuple, Type, Union
import warnings
from attrs import define, field
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
import numpy as np

from torch import dropout, nn
import torch
import torchvision

from rblur.model_utils import _make_first_dim_last, _make_last_dim_first, merge_strings, mm, str_to_act_and_dact_fn, _compute_conv_output_shape
from rblur.supconloss import SupConLoss, AngularSupConLoss
from fastai.vision.models.xresnet import xresnet34, xresnet18, xresnet50, XResNet
from fastai.layers import ResBlock
from einops import rearrange
from fastai.layers import ResBlock
from rblur.cornet_s import CORnet_S

from rblur.wide_resnet import Wide_ResNet
from rblur.runners import load_params_into_model
from matplotlib import pyplot as plt

bnrelu = lambda c : nn.Sequential(nn.BatchNorm2d(c),nn.ReLU())
convbnrelu = lambda ci, co, k, s, p: nn.Sequential(nn.Conv2d(ci, co, k, s, p),bnrelu(co))

@define(slots=False)
class CommonModelParams:
    input_size: Union[int, List[int]] = None
    num_units: Union[int, List[int]] = None
    activation: Type[nn.Module] = nn.ReLU
    bias: bool = True
    dropout_p: float = 0.

@define(slots=False)
class ClassificationModelParams:
    num_classes: int = 1

class CommonModelMixin(object):
    def add_common_params_to_name(self):
        activation_str = self.activation.__str__()
        self.name +=f'-{activation_str[:activation_str.index("(")]}'

        if not self.use_bias:
            self.name += '-noBias'

        if self.sparsify_act:
            self.name += f'-{self.sparsity_coeff:.3f}Sparse'

    def load_common_params(self) -> None:
        input_size = self.params.common_params.input_size
        num_units = self.params.common_params.num_units
        activation = self.params.common_params.activation
        bias = self.params.common_params.bias

        self.input_size = input_size
        self.num_units = num_units
        self.activation = activation
        self.use_bias = bias
        self.dropout_p = self.params.common_params.dropout_p

class ClassificationModelMixin(object):
    def load_classification_params(self) -> None:
        self.num_classes = self.params.classification_params.num_classes
        self.num_logits = (1 if self.num_classes == 2 else self.num_classes)


class IdentityLayer(AbstractModel):
    def forward(self, x):
        return x

    def compute_loss(self, x, y, return_logits=True):
        logits = x
        loss = torch.zeros((x.shape[0],), device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss    

class ActivationLayer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        activation_cls: Type[nn.Module] = None
    
    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self.activation = self.params.activation_cls()

    def forward(self, x):
        return self.activation(x)

    def compute_loss(self, x, y, return_logits=True):
        logits = x
        loss = torch.zeros((x.shape[0],), device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

class BatchNorm2DLayer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        num_features: int = None
        eps: float = 1e-05
        momentum: float = 0.1
        affine: bool = True
        track_running_stats: bool = True
    
    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self.bn = nn.BatchNorm2d(params.num_features, params.eps, params.momentum, params.affine, params.track_running_stats)
    
    def forward(self, x):
        return self.bn(x)

    def compute_loss(self, x, y, return_logits=True):
        logits = x
        loss = torch.zeros((x.shape[0],), device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

    
class SequentialLayers(AbstractModel, CommonModelMixin):
    @define(slots=False)
    class ModelParams(BaseParameters):
        layer_params: List[BaseParameters] = field(factory=list)
        common_params: CommonModelParams = field(factory=CommonModelParams)
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self.load_common_params()
        self._make_network()
        self._make_name()
    
    def _make_name(self):
        self.name = merge_strings([l.name for l in self.layers])
        self.name += f'-{self.params.common_params.dropout_p}Dropout'
    
    def _make_network(self):
        if self.params.common_params.input_size is not None:
            input_size = self.params.common_params.input_size
        else:
            input_size = self.params.layer_params[0].common_params.input_size
        x = torch.rand(1,*input_size)
        layers = []
        print(x.shape)
        for lp in self.params.layer_params:
            if hasattr(lp, 'common_params'):
                lp.common_params.input_size = x.shape[1:]
            l = lp.cls(lp)
            x = l(x)
            if isinstance(x, tuple):
                x = x[0]
            print(type(l), x.shape)
            layers.append(l)
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, *fwd_args, **fwd_kwargs):
        out = x
        extra_outputs = []
        for l in self.layers:
            if self.training:
                out = l.compute_loss(out, None, return_logits=True)
            else:
                out = l(out, *fwd_args, **fwd_kwargs)
            if isinstance(out, tuple):
                extra_outputs.append(out[1:])
                out = out[0]
        if len(extra_outputs) > 0:
            return out, extra_outputs
        else:
            return out
    
    def compute_loss(self, x, y, return_logits=True, **fwd_kwargs):
        out = self.forward(x, **fwd_kwargs)
        if isinstance(out, tuple):
            logits = out[0]
            loss = out[1][0]
        else:
            logits = out
            loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

class GeneralClassifier(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        input_size: List[int] = None
        feature_model_params: BaseParameters = None
        classifier_params: BaseParameters = None
        logit_ensembler_params: BaseParameters = None
        loss_fn: nn.Module = nn.CrossEntropyLoss
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self._make_network()
        self._make_name()
    
    def _make_name(self):
        self.name = self.feature_model.name
    
    def _make_network(self):
        fe_params = self.params.feature_model_params
        self.feature_model = fe_params.cls(fe_params)
        input_size = self.params.input_size if self.params.input_size is not None else fe_params.common_params.input_size
        x = torch.rand(1,*(input_size))
        x = self.feature_model(x)
        if isinstance(x, tuple):
            x = x[0]

        cls_params = self.params.classifier_params
        if hasattr(cls_params, 'common_params'):
            cls_params.common_params.input_size = x.shape[1:]
        else:
            cls_params.input_size = x.shape[1:]
        self.classifier = cls_params.cls(cls_params)
        if self.params.logit_ensembler_params is not None:
            self.logit_ensembler = self.params.logit_ensembler_params.cls(self.params.logit_ensembler_params)
        else:
            self.logit_ensembler = nn.Identity()
        self.loss_fn = self.params.loss_fn()

    def _get_feats(self, x):
        r = self.feature_model.forward(x)
        if isinstance(r, tuple):
            r = r[0]
        return r
    
    def _run_classifier(self, x):
        logits = self.classifier(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = self.logit_ensembler(logits)
        return logits
    
    def forward(self, x, *fwd_args, **fwd_kwargs):
        y = self.feature_model.forward(x, *fwd_args, **fwd_kwargs)
        extra_outputs = []
        if isinstance(y, tuple):
            r = y[0]
            conv_extra_outputs = y[1]
            extra_outputs.extend(conv_extra_outputs)
        else:
            r = y
            conv_extra_outputs = ()
        # if np.random.uniform(0, 1) < 1/100:
        #     _r = r.detach().cpu().numpy()
        #     print(f'sparsity={(_r == 0).astype(float).mean():3f}', _r.max(), np.quantile(_r, 0.99))
        out = self.classifier(r, *fwd_args, **fwd_kwargs)
        if isinstance(out, tuple):
            logits = out[0]
            cls_extra_outputs = out[1:]
            extra_outputs.append(cls_extra_outputs)
        else:
            logits = out
            cls_extra_outputs = ()
        logits = self.logit_ensembler(logits)
        if len(cls_extra_outputs) == 0 and len(conv_extra_outputs) == 0:
            return logits
        extra_outputs = list(zip(*extra_outputs))
        return (logits, *extra_outputs)

    def compute_loss(self, x, y, *fwd_args, return_logits=True, **fwd_kwargs):
        out = self.forward(x, *fwd_args, **fwd_kwargs)
        if isinstance(out, tuple):
            logits = out[0]
            extra_outputs = out[1:]
        else:
            logits = out
            extra_outputs = ()
        # if (logits.dim() == 1) or ((logits.dim() > 1) and (logits.shape[-1] == 1)):
        #     loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        # else:
        #     loss = nn.functional.cross_entropy(logits, y)
        loss = self.loss_fn(logits, y)
        if len(extra_outputs) > 0:
            loss = loss #+ torch.stack(extra_outputs[0], 0).sum(0)
            extra_outputs = extra_outputs[1:]
            assert loss.dim() == 0
        outputs = []
        if return_logits:
            outputs.append(logits)
        outputs.append(loss)
        if len(extra_outputs) > 0:
            outputs.extend(extra_outputs)
        return tuple(outputs)        

@define(slots=False)
class ConvParams:
    kernel_sizes: List[int] = None
    strides: List[int] = None
    padding: List[int] = None

class ConvEncoder(AbstractModel, CommonModelMixin):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        conv_params: ConvParams = field(factory=ConvParams)

    def __init__(self, params: ModelParams) -> None:
        super(ConvEncoder, self).__init__(params)
        self.params = params
        self.load_common_params()
        self._load_conv_params()
        self._make_name()
        self._make_network()

    def _load_conv_params(self):
        self.kernel_sizes = self.params.conv_params.kernel_sizes
        self.strides = self.params.conv_params.strides
        self.padding = self.params.conv_params.padding
    
    def _make_name(self):
        layer_desc = [f'{f}x{ks}_{s}_{p}' for ks,s,p,f in zip(self.kernel_sizes, self.strides, self.padding, self.num_units)]
        layer_desc_2 = []
        curr_desc = ''
        for i,x in enumerate(layer_desc):
            if x == curr_desc:
                count += 1
            else:
                if curr_desc != '':
                    layer_desc_2.append(f'{count}x{curr_desc}')
                count = 1
                curr_desc = x
            if i == len(layer_desc)-1:
                layer_desc_2.append(f'{count}x{curr_desc}')
        self.name = 'Conv-'+'_'.join(layer_desc_2)
    
    def _make_network(self):
        layers = []
        nfilters = [self.input_size[0], *self.num_units]
        kernel_sizes = [None] + self.kernel_sizes
        strides = [None] + self.strides
        padding = [None] + self.padding
        for i, (k,s,f,p) in enumerate(zip(kernel_sizes, strides, nfilters, padding)):
            if i > 0:
                layers.append(nn.Conv2d(nfilters[i-1], f, k, s, p, bias=self.use_bias))
                layers.append(self.activation())
                if self.dropout_p > 0:
                    layers.append(nn.Dropout2d(self.dropout_p))
        self.conv_model = nn.Sequential(*layers)
    
    def forward(self, x, return_state_hist=False, **kwargs):
        feat = self.conv_model(x)
        return feat
    
    def compute_loss(self, x, y, return_state_hist=False, return_logits=False):
        logits = self.forward(x, return_state_hist=True)
        loss = torch.tensor(0., device=x.device)
        output = (loss,)
        if return_state_hist:
            output = output + (None,)
        if return_logits:
            output = (logits,) + output
        return output

class SupervisedContrastiveTrainingWrapper(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        model_params: BaseParameters = None
        projection_params: BaseParameters = None
        use_angular_supcon_loss: bool = False
        angular_supcon_loss_margin: float = 1.
        supcon_loss_temperature: float = 0.07
    
    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self._make_network()

    def _make_network(self):
        super()._make_network()
        self.base_model = self.params.model_params.cls(self.params.model_params)
        self.proj = self.params.projection_params.cls(self.params.projection_params)
        if self.params.use_angular_supcon_loss:
            self.lossfn = AngularSupConLoss(base_temperature=self.params.supcon_loss_temperature, temperature=self.params.supcon_loss_temperature, margin=self.params.angular_supcon_loss_margin)
        else:
            self.lossfn = SupConLoss(base_temperature=self.params.supcon_loss_temperature, temperature=self.params.supcon_loss_temperature)
    
    def _get_proj(self, z):
        if not isinstance(self.proj, IdentityLayer):
            p = self.proj(z)
            p = nn.functional.normalize(p)
        else:
            p = z
        return p
    
    def _get_feats(self, x):
        z = self.base_model._get_feats(x)
        z = nn.functional.normalize(z)
        return z
    
    def _run_classifier(self, x):
        return self.base_model._run_classifier(x)
    
    def forward(self, x):
        return self._run_classifier(self._get_feats(x)) 
    
    def compute_loss(self, x, y, return_logits=True):
        if x.dim() == 5:
            x = rearrange(x, 'b n c h w -> (n b) c h w')
        feat = self._get_feats(x)
        proj = self._get_proj(feat)
        logits = self._run_classifier(feat.detach())

        proj = rearrange(proj, '(n b) d -> b n d', b=y.shape[0])
        logits = rearrange(logits, '(n b) d -> b n d', b=y.shape[0])

        loss1 = self.lossfn(proj, y)
        
        loss2 = 0
        for i in range(logits.shape[1]):
            loss2 += nn.functional.cross_entropy(logits[:, i], y)
        loss2 /= logits.shape[1]
        loss = loss1 + loss2

        logits = logits.mean(1)
        if return_logits:
            return logits, loss
        else:
            return loss

class XResNet34(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        normalization_layer_params: BaseParameters = None
        preprocessing_layer_params: BaseParameters = None
        logit_ensembler_params: BaseParameters = None
        feature_ensembler_params: BaseParameters = None
        setup_feature_extraction: bool = False
        setup_classification: bool = True
        num_classes: int = None
        kernel_size: int = 3
        widen_factor: int = 1.
        widen_stem: bool = False
        stem_sizes: Tuple[int,int,int] = (32,32,64)
        drop_layers: List[int] = []

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params: XResNet34.ModelParams = params
        self._make_network()

    def _make_resnet(self):
        resnet = xresnet34(p=self.params.common_params.dropout_p,
                                c_in=self.params.common_params.input_size[0],
                                n_out=self.params.common_params.num_units,
                                act_cls=self.params.common_params.activation,
                                widen=self.params.widen_factor,
                                stem_szs= self.params.stem_sizes if not self.params.widen_stem else (self.params.widen_factor*self.params.stem_sizes[0],self.params.widen_factor*self.params.stem_sizes[1],self.params.stem_sizes[2]),
                                ks=self.params.kernel_size
                            )
        for i in self.params.drop_layers:
            resnet[i] = nn.Identity()
        resnet[-1] = nn.Identity()
        return resnet

    def _make_network(self):
        if self.params.normalization_layer_params is not None:
            self.normalization_layer = self.params.normalization_layer_params.cls(self.params.normalization_layer_params)
        else:
            self.normalization_layer = nn.Identity()
        if self.params.preprocessing_layer_params is not None:
            self.preprocessing_layer = self.params.preprocessing_layer_params.cls(self.params.preprocessing_layer_params)
        else:
            self.preprocessing_layer = nn.Identity()
        self.resnet = self._make_resnet()
        if self.params.setup_classification:
            x = self.resnet(torch.rand(2, *(self.params.common_params.input_size)))
            self.classifier = nn.Linear(x.shape[1], self.params.num_classes)
        if self.params.logit_ensembler_params is not None:
            self.logit_ensembler = self.params.logit_ensembler_params.cls(self.params.logit_ensembler_params)
        else:
            self.logit_ensembler = nn.Identity()
        if self.params.feature_ensembler_params is not None:
            self.feature_ensembler = self.params.feature_ensembler_params.cls(self.params.feature_ensembler_params)
        else:
            self.feature_ensembler = nn.Identity()

    def _get_feats(self, x, **kwargs):
        x = self.normalization_layer(x)
        x = self.preprocessing_layer(x)
        feat = self.resnet(x)
        feat = self.feature_ensembler(feat)
        return feat
    
    def _run_classifier(self, x):
        logits = self.classifier(x)
        logits = self.logit_ensembler(logits)
        return logits
    
    def forward(self, x, **kwargs):
        x = self._get_feats(x)
        if self.params.setup_classification:
            x =  self._run_classifier(x)
        return x
    
    def compute_loss(self, x, y, return_logits=True, **kwargs):
        logits = self.forward(x)
        if self.params.setup_classification:
            loss = nn.functional.cross_entropy(logits, y)
        else:
            loss = torch.tensor(0., device=x.device)
        # if self.training:
        #     print(logits[:2], loss)
        output = (loss,)
        if return_logits:
            output = (logits,) + output
        return output

class XResNet18(XResNet34):
    def _make_resnet(self):
        resnet = xresnet18(p=self.params.common_params.dropout_p,
                                c_in=self.params.common_params.input_size[0],
                                n_out=self.params.common_params.num_units,
                                act_cls=self.params.common_params.activation,
                                widen=self.params.widen_factor,
                                stem_szs= self.params.stem_sizes if not self.params.widen_stem else (self.params.widen_factor*self.params.stem_sizes[0],self.params.widen_factor*self.params.stem_sizes[1],self.params.stem_sizes[2]),
                                ks=self.params.kernel_size,
                            )
        for i in self.params.drop_layers:
            resnet[i] = nn.Identity()
        resnet[-1] = nn.Identity()
        return resnet

class XResNet50(XResNet34):
    def _make_resnet(self):
        resnet = xresnet50(p=self.params.common_params.dropout_p,
                                c_in=self.params.common_params.input_size[0],
                                n_out=self.params.common_params.num_units,
                                act_cls=self.params.common_params.activation,
                                widen=self.params.widen_factor,
                                stem_szs= self.params.stem_sizes if not self.params.widen_stem else (self.params.widen_factor*self.params.stem_sizes[0],self.params.widen_factor*self.params.stem_sizes[1],self.params.stem_sizes[2]),
                                ks=self.params.kernel_size
                            )
        for i in self.params.drop_layers:
            resnet[i] = nn.Identity()
        resnet[-1] = nn.Identity()
        return resnet

class CORnetS(XResNet34):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        normalization_layer_params: BaseParameters = None
        setup_feature_extraction: bool = False
        setup_classification: bool = True
        num_classes: int = None
        num_recurrence: List[int] = [2,4,2]

    def _make_resnet(self):
        resnet = CORnet_S(times=self.params.num_recurrence)
        return resnet

class WideResnet(XResNet34):
    @define(slots=False)
    class ModelParams(XResNet34.ModelParams):
        depth: int = None
        widen_factor: int = None
    
    def _make_resnet(self):
        resnet = Wide_ResNet(self.params.depth, self.params.widen_factor, self.params.common_params.dropout_p, self.params.num_classes)
        resnet.linear = nn.Identity()
        return resnet

class LogitAverageEnsembler(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        n: int = None
        activation: nn.Module = nn.Identity
        reduction: str = 'mean'
    
    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self.activation = self.params.activation()

    def __repr__(self):
        return f'LogitAverageEnsembler(n={self.params.n}, act={self.activation})'

    def forward(self, x):
        # Assumes that all the logits from the n instances are consecuitve i.e.
        # x = x_.reshape(-1, C) where x_.shape = [b, n, C]
        bn, c = x.shape
        x = self.activation(x)
        if (bn < self.params.n) or (bn % self.params.n != 0):
            w = f'Expected the size of x at dim 0 to be a non-zero multiple of {self.params.n} but got {bn}. Returning x as is.'
            warnings.warn(w)
            return x
        else:
            x = rearrange(x, '(b n) c -> b n c', n=self.params.n)
        if self.params.reduction == 'mean':
            x = x.mean(1)
        elif self.params.reduction == 'logsumexp':
            x = torch.logsumexp(x, 1)
        elif self.params.reduction == 'log_softmax+logsumexp':
            x = nn.functional.log_softmax(x, -1)
            x = torch.logsumexp(x, 1)
        return x
    
    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        output = (loss,)
        if return_logits:
            output = (logits,) + output
        return output
    
class MultiheadSelfAttentionEnsembler(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        n: int = None
        embed_dim: int = None
        num_heads: int = None
        dropout: float =0.
        bias: bool = True
        add_bias_kv: bool = False
        add_zero_attn: bool = False
        kdim: int = None
        vdim: int = None
        batch_first: bool = True
        n_layers: int = 1
        n_repeats: int = 1

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self._make_network()

    def _make_network(self):
        keys_to_exclude = set(['cls', 'n', 'n_layers', 'n_repeats'])
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(**(self.params.asdict(filter=lambda a,v: a.name not in keys_to_exclude))) for i in range(self.params.n_layers)
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.params.embed_dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.params.n + 1, self.params.embed_dim))
        self.intermediate = nn.ModuleList([nn.Sequential(
            nn.Linear(self.params.embed_dim, self.params.embed_dim),
            nn.ReLU()
        ) for i in range(self.params.n_layers)])
        self.output = nn.ModuleList([nn.Linear(self.params.embed_dim, self.params.embed_dim) for i in range(self.params.n_layers)])
        self.layernorm_1 = nn.ModuleList([nn.LayerNorm(self.params.embed_dim) for i in range(self.params.n_layers)])
        self.layernorm_2 = nn.ModuleList([nn.LayerNorm(self.params.embed_dim) for i in range(self.params.n_layers)])
    
    def forward(self, x_inp):
        x = x_inp.flatten(1)
        # assumes that the batch dimension is flattened from b x n,
        # where n is seq len
        x = x.flatten(1).reshape(-1, self.params.n, x.shape[-1])
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        for i, (layer, interm, output, lnorm1, lnorm2) in enumerate(zip(
                self.attn_layers,
                self.intermediate,
                self.output,
                self.layernorm_1,
                self.layernorm_2
            )):
            for j in range(self.params.n_repeats):
                x_norm = lnorm1(x)
                x_att, _ = layer(x_norm, x_norm, x_norm)
                if (i == (self.params.n_layers-1)) and (j == (self.params.n_repeats - 1)):
                    x, x_att = x[:, 0], x_att[:, 0]
                x = x + x_att
                x_int = interm(lnorm2(x))
                x = output(x_int) + x
        # avg_x = x_inp.flatten(1).reshape(-1, self.params.n, x.shape[-1]).mean(1)
        # print(torch.norm(x - avg_x, dim=1).mean())
        return x

class LSTMEnsembler(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        n: int = None
        input_size: int = None
        hidden_size: int = None
        num_layers: int = 1
        bias: bool = True
        batch_first: bool = True
        dropout: float = 0.
        bidirectional: bool = False
        proj_size: int = 0
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self._make_network()

    def _make_network(self):
        keys_to_exclude = set(['cls', 'n'])
        lstm_kwargs = self.params.asdict(filter=lambda a,v: a.name not in keys_to_exclude)
        self.lstm = nn.LSTM(**lstm_kwargs)
        h_out = self.params.proj_size if (self.params.proj_size > 0) else self.params.hidden_size
        d = 2 if self.params.bidirectional else 1
        self.h0 = nn.parameter.Parameter(torch.zeros(d*self.params.num_layers, self.params.hidden_size))
        self.c0 = nn.parameter.Parameter(torch.zeros(d*self.params.num_layers, self.params.hidden_size))
    
    def forward(self, x_inp):
        # assumes that the batch dimension is flattened from b x n,
        # where n is seq len
        x = x_inp.flatten(1).reshape(-1, self.params.n, self.params.input_size)
        h0 = self.h0.unsqueeze(1).repeat_interleave(x.shape[0], 1)
        c0 = self.c0.unsqueeze(1).repeat_interleave(x.shape[0], 1)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        last_output = output[:, -1]
        return last_output

class NormalizationLayer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        mean: Union[float, List[float]] = [0.485, 0.456, 0.406]
        std: Union[float, List[float]] = [0.229, 0.224, 0.225]

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        if isinstance(self.params.mean, list):
            self.mean = nn.parameter.Parameter(torch.FloatTensor(self.params.mean).reshape(1,-1,1,1), requires_grad=False)
        if isinstance(self.params.std, list):
            self.std = nn.parameter.Parameter(torch.FloatTensor(self.params.std).reshape(1,-1,1,1), requires_grad=False)
    
    def __repr__(self):
        return f'NormalizationLayer(mean={self.params.mean}, std={self.params.std})'
    
    def forward(self, x):
        return (x-self.mean)/self.std
    
    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = torch.zeros((logits.shape[0],), device=x.device)
        return logits
    
class PretrainedTimmModel(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        model_name: str = None
    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        import timm
        self.model = timm.create_model(params.model_name, pretrained=True)
        pretrained_cfg = self.model.pretrained_cfg
        self.norm_layer = NormalizationLayer(NormalizationLayer.ModelParams(
            NormalizationLayer,
            mean=list(pretrained_cfg.get('mean', [0.]*3)),
            std=list(pretrained_cfg.get('std', [1.]*3)),
        ))

    def forward(self, x):
        x = self.norm_layer(x)
        return self.model(x)

    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        output = (loss,)
        if return_logits:
            output = (logits,) + output
        return output

class FovTexVGG(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        scale: str = '0.4'
        num_classes: int = None
        permutation: str = None

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()

    def _make_network(self):
        from rblur.FoveatedTextureTransform.model_arch import vgg11_tex_fov
        self.vgg = vgg11_tex_fov(self.params.scale, self.params.common_params.input_size[1], 
                                    self.params.num_classes, self.params.permutation)

    def forward(self, x):
        return self.vgg(x)
    
    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        output = (loss,)
        if return_logits:
            output = (logits,) + output
        return output

class XResNetClassifierWithReconstructionLoss(XResNet18):
    @define(slots=False)
    class ModelParams(XResNet18.ModelParams):
        preprocessing_params: BaseParameters = None
        recon_wt: float = 1.
        cls_wt: float = 1.
        feature_layer_idx: int = -1
        logit_ensembler_params: BaseParameters = None

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        params.setup_classification = params.setup_feature_extraction = True
        self._make_network()

    def _make_reconstructor(self):
        isz = self.params.common_params.input_size
        shapes = [(1, *isz)]
        isz = shapes[-1]
        for i, block in enumerate(self.resnet):
            if i <= self.params.feature_layer_idx:
                x = torch.rand(*isz)
                x = block(x)
                isz = x.shape
                shapes.append(isz)
        recon_layers = []
        self.combiners = nn.ModuleList([])
        for i in range(len(shapes)-1, 0, -1):
            s = shapes[i-1][2]//shapes[i][2]
            l = nn.ConvTranspose2d(shapes[i][1], shapes[i-1][1], 3, s, 1, int(s > 1))
            # l = nn.Conv2d(shapes[i][1], shapes[i-1][1]*(s**2), 3, 1, 1)
            # if s > 1:
            #     l = nn.Sequential(l, nn.PixelShuffle(s))
            x = l(x)
            recon_layers.append(l)
            if i < len(shapes)-1:
                c = nn.Conv2d(2*shapes[i][1], shapes[i][1], 1, 1, 0)
                self.combiners.append(c)
        self.reconstructor = nn.Sequential(*recon_layers)
    
    def _make_network(self):
        super()._make_network()
        if self.params.preprocessing_params is None:
            self.preprocessor = nn.Identity()
        else:
            self.preprocessor = self.params.preprocessing_params.cls(self.params.preprocessing_params)
        self._make_reconstructor()
        if self.params.logit_ensembler_params is not None:
            self.logit_ensembler = self.params.logit_ensembler_params.cls(self.params.logit_ensembler_params)
        else:
            self.logit_ensembler = nn.Identity()
    
    def preprocess(self, x):
        x = self.preprocessor(x)
        if isinstance(x, tuple):
            x = x[0]
        return x
    
    def _get_feats_(self, x, return_interm_activations=False, **kwargs):
        x = self.normalization_layer(x)
        feats = []
        for i,l in enumerate(self.resnet):
            if i <= self.params.feature_layer_idx:
                x = l(x)
                feats.append(x)
        if return_interm_activations:
            return x, feats
        return x
    
    def _get_feats(self, x, return_interm_activations=False, **kwargs):
        x = self.preprocess(x)
        return self._get_feats_(x, return_interm_activations=return_interm_activations, **kwargs)
    
    def _run_classifier(self, x):
        for i,l in enumerate(self.resnet):
            if i > self.params.feature_layer_idx:
                x = l(x)
        logits = self.classifier(x)
        logits = self.logit_ensembler(logits)
        return logits

    def forward_and_reconstruct(self, x):
        x = self.preprocess(x)
        feats, all_feats = self._get_feats_(x, return_interm_activations=True)
        if self.params.recon_wt > 0:
            all_feats = all_feats[::-1]
            for i,l in enumerate(self.reconstructor):
                f = all_feats[i]
                if i > 0:
                    f = torch.relu(self.combiners[i-1](torch.cat([r, f], 1)))
                r = l(f)
                if i < (len(self.reconstructor)-1):
                    r = torch.relu(r)
        else:
            r = torch.zeros_like(x)
        if self.params.cls_wt > 0:
            logits = self._run_classifier(feats)
        else:
            logits = torch.zeros((x.shape[0], self.params.num_classes), dtype=x.dtype, device=x.device)
        plt.subplot(1,2,1)
        plt.imshow(convert_image_tensor_to_ndarray(x[0]))
        plt.subplot(1,2,2)
        plt.imshow(convert_image_tensor_to_ndarray(r[0]))
        plt.savefig('recon2.png')
        return logits, r
    
    def compute_loss(self, x, y, return_logits=True):
        if (self.params.recon_wt > 0):# and self.training:
            logits, recon = self.forward_and_reconstruct(x)
            rloss = torch.pow(x.reshape(x.shape[0],-1) - recon.reshape(recon.shape[0],-1), 2).mean()
        else:
            logits = self.forward(x)
            rloss = 0
        closs = nn.functional.cross_entropy(logits, y)
        loss = self.params.cls_wt * closs + self.params.recon_wt * rloss
        if return_logits:
            return logits, loss
        return loss

def convert_image_tensor_to_ndarray(img):
    return img.cpu().detach().transpose(0,1).transpose(1,2).numpy()

class XResNetClassifierWithEnhancer(XResNetClassifierWithReconstructionLoss):
    @define(slots=False)
    class ModelParams(XResNetClassifierWithReconstructionLoss.ModelParams):
        no_reconstruction: bool = False
        use_residual_during_inference: bool = False

    def _make_reconstructor(self):
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(3, 32, 9, 4, 4),
            nn.Conv2d(3, 32, 7, 4, 3),
            nn.Conv2d(3, 32, 5, 4, 2),
        ])
        self.bnrelu = nn.Sequential(
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.reconstructor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(96, 64, 5, 1, 2),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(160, 64, 1, 1),
                nn.ReLU(),
                nn.Conv2d(64, 3*16, 5, 1, 2),
                nn.PixelShuffle(4)
            ),
        ])

    def reconstruct(self, x):
        z1 = self.bnrelu(torch.cat([l(x) for l in self.multi_scale_conv],1))
        z2 = self.reconstructor[0](z1)
        # z2 = self.reconstructor[1](z1)
        r = self.reconstructor[1](torch.cat((z1,z2), 1))
        return r
    
    def _get_feats(self, x, return_interm_activations=False, return_reconstruction=False, **kwargs):
        xp = self.preprocess(x)
        if self.params.no_reconstruction:
            r = xp
        else:
            r = self.reconstruct(xp)
        if self.params.use_residual_during_inference and (not self.training):
            x = (x - r)
        # plt.subplot(1,2,1)
        # plt.imshow(convert_image_tensor_to_ndarray(x[0]))
        # plt.subplot(1,2,2)
        # plt.imshow(convert_image_tensor_to_ndarray(r[0]))
        # plt.savefig('recon2.png')
        
        out = self._get_feats_(r, return_interm_activations=return_interm_activations, **kwargs)
        if return_reconstruction:
            if isinstance(out, tuple):
                out = out + (r,)
            else:
                out = (out, r)
        return out
    
    def forward_and_reconstruct(self, x):
        f, r = self._get_feats(x, return_reconstruction=True)
        if self.params.cls_wt > 0:
            logits = self._run_classifier(f)
        else:
            logits = torch.zeros((x.shape[0], self.params.num_classes), dtype=x.dtype, device=x.device)
        return logits, r

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        return out

class DnCNN(nn.Module):
    def __init__(self, num_channels, num_resblocks):
        super().__init__()
        # define the layers of the DnCNN
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_resblocks)]
        )
        self.conv2 = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # apply the convolutional layers and ReLU activation function
        x = torch.relu(self.conv1(x))
        x = self.resblocks(x)
        x = self.conv2(x)
        return x

class XResNetClassifierWithDeepResidualEnhancer(XResNetClassifierWithEnhancer):
    @define(slots=False)
    class ModelParams(XResNetClassifierWithEnhancer.ModelParams):
        perceptual_loss: bool = False
        ploss_cnn_layer_idx: int = 12
        resnet_ckp_path: str = None
        reconstructor_ckp_path: str = None

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        if params.resnet_ckp_path is not None:
            load_params_into_model(torch.load(params.resnet_ckp_path), self, r'(.*convs.*|.*multi_scale_conv.*|.*bnrelu.*|.*upsample.*)')
        if params.reconstructor_ckp_path is not None:
            load_params_into_model(torch.load(params.reconstructor_ckp_path), self, r'(.*resnet.*)')

    def _make_reconstructor(self):
        bnrelu = lambda c : nn.Sequential(nn.BatchNorm2d(c),nn.ReLU())
        convbnrelu = lambda ci, co, k, s, p: nn.Sequential(nn.Conv2d(ci, co, k, s, p),bnrelu(co))
    #     self.reconstructor = nn.Sequential(
    #         convbnrelu(3, 128, 15, 2, 7),
    #         convbnrelu(128, 320, 1, 1, 0),
    #         convbnrelu(320, 320, 1, 1, 0),
    #         convbnrelu(320, 320, 3, 2, 1),
    #         convbnrelu(320, 128, 1, 1, 0),
    #         convbnrelu(128, 128, 3, 1, 1),
    #         convbnrelu(128, 512, 1, 1, 0),
    #         convbnrelu(512, 48*4, 5, 1, 2),
    #         nn.PixelShuffle(2),
    #         convbnrelu(48, 96, 3, 1, 1),
    #         nn.Conv2d(96, 3*4, 5, 1, 2),
    #         nn.PixelShuffle(2)
    #     )
    
    # def reconstruct(self, x):
    #     return self.reconstructor(x)

        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(3, 256, 13, 2, 6),
            # nn.Conv2d(3, 64, 9, 2, 4),
            # nn.Conv2d(3, 64, 7, 2, 3),
            # nn.Conv2d(3, 64, 5, 2, 2),
        ])
        self.bnrelu = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.convs = nn.Sequential(
            convbnrelu(256, 64, 3, 1, 1),
            convbnrelu(64, 64, 3, 1, 1),
            *(8*[convbnrelu(128, 64, 3, 1, 1)]),
        )
        self.upsample = nn.Sequential(
            convbnrelu(128, 32*4, 3, 1, 1),
            nn.PixelShuffle(2),
            convbnrelu(32, 32, 3, 1, 1),
            # nn.PixelShuffle(2),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
        # self.reconstructor = DnCNN(3, 16)
        if self.params.perceptual_loss:
            self.ploss_cnn = torchvision.models.vgg16_bn(True).features[:self.params.ploss_cnn_layer_idx+1].requires_grad_(False)

    def reconstruct(self, xo):
        x = self.bnrelu(torch.cat([l(xo) for l in self.multi_scale_conv],1))
        for i,l in enumerate(self.convs):
            f1 = l(x)
            if i > 0:
                x = torch.cat([f0,f1], 1)
            else:
                x = f1
            f0 = f1
        x = self.upsample(x)
        return x

    def compute_perceptual_loss(self, x, r):
        x = self.normalization_layer(x)
        r = self.normalization_layer(r)
        
        fx = self.ploss_cnn(x)
        fr = self.ploss_cnn(r)
        loss = torch.pow(fx - fr, 2).sum(1).mean()
        return loss

    def compute_loss(self, x, y, return_logits=True):
        if (self.params.recon_wt > 0):# and self.training:
            logits, recon = self.forward_and_reconstruct(x)
            if self.params.perceptual_loss:
                rloss = self.compute_perceptual_loss(x, recon)
            else:
                rloss = torch.pow(x.reshape(x.shape[0],-1) - recon.reshape(recon.shape[0],-1), 2).mean()
        else:
            logits = self.forward(x)
            rloss = 0
        closs = nn.functional.cross_entropy(logits, y)
        loss = self.params.cls_wt * closs + self.params.recon_wt * rloss
        if return_logits:
            return logits, loss
        return loss

class FlattenLayer(AbstractModel):
    def forward(self, x, *args, **kwargs):
        return x.reshape(x.shape[0], -1)

    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = torch.zeros((x.shape[0],), device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

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

class DropoutLayer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        dropout_p: float = 0.
    
    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self.dropout = nn.Dropout(params.dropout_p)

    def forward(self, x, *args, **kwargs):
        return self.dropout(x)

    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = torch.zeros((x.shape[0],), device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

class NormalizationLayer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        mean: Union[float, List[float]] = [0.485, 0.456, 0.406]
        std: Union[float, List[float]] = [0.229, 0.224, 0.225]

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        if isinstance(self.params.mean, list):
            self.mean = nn.parameter.Parameter(torch.FloatTensor(self.params.mean).reshape(1,-1,1,1), requires_grad=False)
        if isinstance(self.params.std, list):
            self.std = nn.parameter.Parameter(torch.FloatTensor(self.params.std).reshape(1,-1,1,1), requires_grad=False)
    
    def __repr__(self):
        return f'NormalizationLayer(mean={self.params.mean}, std={self.params.std})'
    
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