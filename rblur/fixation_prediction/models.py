
from collections import OrderedDict
from typing import Callable, List, Literal, Union
from attrs import define, field
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
import numpy as np

from torch import nn
import torch
from fastai.vision.models.xresnet import xresnet18, XResNet
from fastai.layers import ResBlock
from fastai.layers import ResBlock
from rblur.runners import load_params_into_model
import torchvision
from torchvision.models.segmentation import fcn_resnet50, fcn, deeplabv3_resnet50, deeplabv3
from torchvision.models.resnet import resnet18, ResNet
from torchvision.models._utils import IntermediateLayerGetter
from matplotlib import pyplot as plt
import math
from einops import rearrange

from rblur.models import CommonModelParams, ConvEncoder, XResNet34, convbnrelu, bnrelu
from rblur.retina_blur2 import RetinaBlurFilter as RBlur2
import deepgaze_pytorch
from pathlib import Path

def convert_image_tensor_to_ndarray(img):
    return img.cpu().detach().transpose(0,1).transpose(1,2).numpy()

def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    copied from https://github.com/pytorch/pytorch/issues/39242
    """
    mask = -math.log(2) < x  # x < 0
    return torch.where(
        mask,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )

def add_gumbel_noise(logits):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    return gumbels

def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, log: bool = False, dim: int = -1) -> torch.Tensor:
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    if log:
        y_soft = gumbels.log_softmax(dim)
    else:
        y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class SkipConnection(nn.Sequential):
    def __init__(self, *block_layers) -> None:
        super().__init__(*block_layers)
    
    def forward(self, x):
        return x + super().forward(x)

class UNet(nn.Module):
    def __init__(self, num_filters=[64,128,256,512,1024]) -> None:
        super().__init__()
        ks = 3
        pd = ks//2
        dsblock = lambda ic, oc: nn.Sequential(
            convbnrelu(ic, oc, ks, 2, pd),
            SkipConnection(
                convbnrelu(oc, oc, ks, 1, pd),
                convbnrelu(oc, oc, ks, 1, pd)
            ),
        )
        usblock = lambda ic, hc, oc: nn.Sequential(
            convbnrelu(ic, hc, ks, 1, pd),
            SkipConnection(
                convbnrelu(hc, hc, ks, 1, pd)
            ),
            nn.ConvTranspose2d(hc, oc, ks, 2, pd, 1),
            bnrelu(oc),
        )
        [f1,f2,f3,f4,f5] = num_filters
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                convbnrelu(3, f1, ks, 1, pd),
                # convbnrelu(f1, f1, ks, 1, pd),
                deeplabv3.ASPP(f1, [6, 12, 18], f1),
            ),
            dsblock(f1, f2),
            dsblock(f2, f3),
            dsblock(f3, f4),
            dsblock(f4, f5),
        ])
        self.upsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(f5, f4, ks, 2, pd, 1),
                bnrelu(f4),
            ),
            usblock(f5, f4, f3),
            usblock(f4, f3, f2),
            usblock(f3, f2, f1),
        ])
        self.mask_predictor = nn.Sequential(
            convbnrelu(f2, f1, ks, 1, pd),
            convbnrelu(f1, f1, ks, 1, pd),
            nn.Conv2d(f1, 1, 1, 1)
        )
        # self.aspp_head = deeplabv3.DeepLabHead(f5, f5)
    
    def forward(self, x):
        dsfeats = []
        for l in self.downsample_layers:
            x = l(x)
            dsfeats.append(x)

        dsfeats = dsfeats[:-1][::-1]
        # x = self.aspp_head(x)
        # dsfeats.append(None)
        assert len(dsfeats) == len(self.upsample_layers)
        for i, (dsf, l) in enumerate(zip(dsfeats, self.upsample_layers)):
            x = l(x)
            x = torch.cat([x, dsf], 1)
        x = self.mask_predictor(x)
        return x

class GALA(nn.Module):
    def __init__(self, backbone, in_channels=256, shrink_factor=4) -> None:
        super().__init__()
        self.backbone = backbone
        self.global_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels, in_channels//shrink_factor, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//shrink_factor, in_channels, 1),
        )
        self.local_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//shrink_factor, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//shrink_factor, 1, 1),
        )
        self.integrator = nn.Conv2d(in_channels*2,1,1)
    
    def forward(self, x):
        input_size = x.shape
        x = self.backbone(x)['out']
        G = self.global_attn(x)
        S = self.local_attn(x)
        H = torch.cat([G+S, G*S], 1)
        A = self.integrator(H)
        A = nn.functional.interpolate(A, size=input_size[2:], mode='bilinear', align_corners=False)
        return A

class DeepLab3p(nn.Module):
    def __init__(self, backbone, nclasses, hl_channels=512, ll_channels=64, upsample_output=True) -> None:
        super().__init__()
        self.backbone = backbone
        self.aspp_layer = deeplabv3.ASPP(hl_channels, [1, 6, 12, 18], 256)
        self.decoder_conv1 = convbnrelu(ll_channels, 64, 1, 1, 0)
        self.decoder = nn.Sequential(
            convbnrelu(256+64, 256, 3, 1, 1),
            nn.Dropout(0.5),
            convbnrelu(256, 256, 3, 1, 1),
            nn.Dropout(0.1),
            nn.Conv2d(256, nclasses, 1, 1)
        )
        self.upsample_output = upsample_output
    
    def forward(self, x):
        input_size = x.shape
        out = self.backbone(x)
        x = out['out']
        x = self.aspp_layer(x)

        llfeat = self.decoder_conv1(out['low_level_feats'])
        x = nn.functional.interpolate(x, size=llfeat.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, llfeat], 1)
        x = self.decoder(x)
        if self.upsample_output:
            x = nn.functional.interpolate(x, size=input_size[2:], mode='bilinear', align_corners=False)
        return x

class BaseFixationPredictor(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        backbone_params: BaseParameters = None
        backbone_config: dict = []
        random_fixation_prob: float = 0.
        loc_sampling_temperature: float = 1.
        mask_past_fixations: bool = True
        always_recompute_fmap: bool = False
        pretrained: bool = True
        min_image_dim: int = 768
        fixation_width_frac: float = 0.1

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        print(params)
        self.random_fixation_prob = params.random_fixation_prob
        self.loc_sampling_temperature = params.loc_sampling_temperature
        self.centerbias_template = nn.parameter.Parameter(torch.FloatTensor(np.load(Path.home() / 'projects/adversarialML/biologically_inspired_models/DeepGaze/centerbias_mit1003.npy')), requires_grad=False)
        self.mask_past_fixations = params.mask_past_fixations
        self.last_input_image = None
        self.last_fixation_map = None
        self.always_recompute_fmap = params.always_recompute_fmap
        self.min_image_dim = params.min_image_dim
        self._make_network()
        self.reset_history()
    
    def _make_network(self):
        self.fixation_predictor = None
    
    def reset_history(self):
        self.hist: torch.Tensor = torch.zeros(1,4, device=self.centerbias_template.device, dtype=self.centerbias_template.dtype)# + 294912
        self.hist[:, :3] += torch.nan
    
    def update_history(self, last_fixation_point: torch.Tensor):
        self.hist = torch.cat((self.hist[:, 1:], last_fixation_point.unsqueeze(1)), 1)
    
    def _update_mask(self, fi, I):
        hks = self.mask_kernel.shape[-1] // 2
        fr, fc = (fi // self.mask.shape[3], fi % self.mask.shape[3])
        top, bot, left, right = fr-hks, fr+hks+1, fc-hks, fc+hks+1
        ktop, kleft = max(0, -top), max(0, -left)
        kbot = self.mask_kernel.shape[0] - max(0, bot - self.mask.shape[2])
        kright = self.mask_kernel.shape[1] - max(0, right - self.mask.shape[3])
        M = self.mask[I]
        M[..., max(0,top):bot, max(0,left):right] *= self.mask_kernel[ktop:kbot, kleft:kright]
        self.mask[I] = M
    
    def update_mask(self, fidx):
        if not self.mask_past_fixations:
            return
        fidx_set = set(fidx.long().cpu().detach().numpy().tolist())
        for fi in fidx_set:
            self._update_mask(fi, (fidx == fi))
    
    def create_mask(self, fmap):
        def gkern(kernlen=256, std=128):
            def gaussian_fn(M, std):
                n = torch.cat([torch.arange(M//2,-1,-1), torch.arange(1, M//2+1)])
                # w = torch.exp(-0.5*((n/std)**2))/(np.sqrt(2*np.pi)*std)
                sig2 = 2 * std * std
                w = torch.exp(-n ** 2 / sig2)
                return w
            assert kernlen%2 == 1
            """Returns a 2D Gaussian kernel array."""
            gkern1d = gaussian_fn(kernlen, std=std) 
            gkern2d = torch.outer(gkern1d, gkern1d)
            return gkern2d
        
        n,_,h,w = fmap.shape
        std = max(h,w) * self.params.fixation_width_frac / 4
        ks = int(4*std)
        ks += 1 - (ks % 2)
        self.mask = torch.ones((n,1,h,w), device=fmap.device)
        if not hasattr(self, 'mask_kernel'):
            self.mask_kernel = 1-gkern(ks, std) + 1e-8
        self.mask_kernel = self.mask_kernel.to(fmap.device)

    def preprocess_image_and_center_bias(self, image):
        sf = self.min_image_dim / min(image.shape[2:])        
        # scale and resize image and center bias to match training regime
        image = image * 255
        if sf != 1:
            image = torch.nn.functional.interpolate(image, scale_factor=sf)
        centerbias = torch.nn.functional.interpolate(self.centerbias_template.unsqueeze(0).unsqueeze(0),
                                                    size=(image.shape[2], image.shape[3]),
                                                    mode='nearest').squeeze(1)
        # normalize the centerbias
        centerbias -= torch.logsumexp(centerbias.reshape(-1), 0)
        return image, centerbias
    
    def postprocess_fixation_map(self, fixation_map, orig_image_size):
        b,c,h,w = fixation_map.shape
        sf = min(orig_image_size) / min(fixation_map.shape[2:])
        if sf != 1:
            fixation_map = torch.nn.functional.interpolate(fixation_map, scale_factor=sf)
        fixation_map += self.mask.log()
        return fixation_map
    
    def predict_fixation_map(self, image, centerbias):
        raise NotImplementedError
    
    def sample_fixation(self, fmap, loc_sampling_temp=1.):
        if self.training:
            masked_flat_prob_map = gumbel_softmax(torch.flatten(fmap, 1), tau=loc_sampling_temp, hard=True)
        else:
            soft_masked_flat_prob_map = torch.softmax(torch.flatten(fmap, 1), 1)
            index = soft_masked_flat_prob_map.max(1, keepdim=True)[1]
            hard_masked_flat_prob_map = torch.zeros_like(soft_masked_flat_prob_map, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
            masked_flat_prob_map = hard_masked_flat_prob_map - soft_masked_flat_prob_map.detach() + soft_masked_flat_prob_map
            # fidx_ds = masked_flat_prob_map.argmax(1)

        index_tensor = torch.arange(masked_flat_prob_map.shape[1], dtype=masked_flat_prob_map.dtype, device=masked_flat_prob_map.device).unsqueeze(1)
        fidx = masked_flat_prob_map.mm(index_tensor).squeeze(-1).int()
        
        rand_fidx = torch.randint_like(fidx, masked_flat_prob_map.shape[1])
        rand_idx = (torch.rand(fidx.shape[0]) < self.random_fixation_prob)
        fidx[rand_idx] = rand_fidx[rand_idx]
        return fidx
    
    def forward(self, image, last_fixation_point=None):
        b,c,h,w = image.shape
        # Initialize or update fixation history
        if last_fixation_point is None:
            self.reset_history()
            self.create_mask(image)
            self.hist = self.hist.repeat_interleave(image.shape[0], 0)
            self.last_fixation_map = self.last_image = None
        else:
            self.update_history(last_fixation_point)
        
        self.update_mask(self.hist.data[:,-1])
        
        if (not self.always_recompute_fmap) and (self.last_fixation_map is not None) and (self.last_image == image).cpu().detach().numpy().all():
            fixation_map = self.last_fixation_map
        else:
            image_scaled, centerbias = self.preprocess_image_and_center_bias(image)
            fixation_map = self.predict_fixation_map(image_scaled, centerbias)
        
        if not self.always_recompute_fmap:
            self.last_fixation_map = fixation_map
            self.last_image = image
        
        fixation_map = self.postprocess_fixation_map(fixation_map, (h,w))

        # get x and y from index
        # x_hist = self.hist % w
        # y_hist = self.hist // w
        # print(x_hist.cpu().detach().numpy(), y_hist.cpu().detach().numpy())
        # b = min(b, 3)
        # f, axs = plt.subplots(max(b,2),2)
        # for i in range(b):
        #     axs[i, 0].imshow(convert_image_tensor_to_ndarray(image[i]))
        #     axs[i, 1].matshow(fixation_map[i,0].exp().cpu().detach().numpy())
        #     axs[i, 1].plot(x_hist[i].cpu().detach().numpy(), y_hist[i].cpu().detach().numpy(), 'o', color='red')
        return fixation_map
    
    def compute_loss(self, image, target_fmap, return_logits=True):
        fmap = self.forward(image)
        fixation_count = target_fmap.sum(dim=(-1, -2), keepdim=True)
        ll = torch.mean(
            torch.sum(fmap * target_fmap, dim=(-1, -2), keepdim=True) / fixation_count
        )
        loss = (ll + np.log(fmap.shape[-1] * fmap.shape[-2])) / np.log(2)

        if return_logits:
            return fmap, loss
        else:
            return loss



class DeepGazeIII(BaseFixationPredictor):
    def __init__(self, params: BaseFixationPredictor.ModelParams) -> None:
        params.always_recompute_fmap = True
        super().__init__(params)
    
    def _make_network(self):
        self.fixation_predictor = deepgaze_pytorch.DeepGazeIII(self.params.pretrained)

    def predict_fixation_map(self, image, centerbias):
        b,c,h,w = image.shape

        # get x and y from index
        x_hist = self.hist % w
        y_hist = self.hist // w
        
        fixation_map = self.fixation_predictor.forward(image, centerbias, x_hist, y_hist, None)
        return fixation_map

class DeepGazeIIIModule(deepgaze_pytorch.modules.DeepGazeIII):
    def forward(self, x, centerbias, x_hist=None, y_hist=None, durations=None, return_features=False):
        orig_shape = x.shape
        x = nn.functional.interpolate(x, scale_factor=1 / self.downsample)
        features = x = self.features(x)

        readout_shape = [math.ceil(orig_shape[2] / self.downsample / self.readout_factor), math.ceil(orig_shape[3] / self.downsample / self.readout_factor)]
        x = [nn.functional.interpolate(item, readout_shape) for item in x]

        x = torch.cat(x, dim=1)
        x = self.saliency_network(x)

        if self.scanpath_network is not None:
            scanpath_features = deepgaze_pytorch.modules.encode_scanpath_features(x_hist, y_hist, size=(orig_shape[2], orig_shape[3]), device=x.device)
            #scanpath_features = F.interpolate(scanpath_features, scale_factor=1 / self.downsample / self.readout_factor)
            scanpath_features = nn.functional.interpolate(scanpath_features, readout_shape)
            y = self.scanpath_network(scanpath_features)
        else:
            y = None

        x = self.fixation_selection_network((x, y))

        x = self.finalizer(x, centerbias)
        if return_features:
            return x, features
        return x

class mFeatureExtractor(deepgaze_pytorch.modules.FeatureExtractor):
    def __init__(self, features, targets):
        super().__init__(features, targets)
        self.input_cache = None
        self.final_output = None

    def forward(self, x, return_model_output=False):
        if self.input_cache is not x:
            self.input_cache = x
            self.outputs.clear()
            self.final_output = self.features(x)
            # plt.figure()
            # plt.imshow(convert_image_tensor_to_ndarray(x[0]))
            # plt.title(f'{self.final_output[0].argmax(-1)}')
        layer_outputs = [self.outputs[target] for target in self.targets]
        if return_model_output:
            return layer_outputs, self.final_output
        return layer_outputs

class CustomBackboneDeepGazeIII(BaseFixationPredictor):
    @define(slots=False)
    class ModelParams(BaseFixationPredictor.ModelParams):
        ckp_path: str = None

    def build_saliency_network(self, input_channels):
        return nn.Sequential(OrderedDict([
            ('layernorm0', deepgaze_pytorch.layers.LayerNorm(input_channels)),
            ('conv0', nn.Conv2d(input_channels, 8, (1, 1), bias=False)),
            ('bias0', deepgaze_pytorch.layers.Bias(8)),
            ('softplus0', nn.Softplus()),

            ('layernorm1', deepgaze_pytorch.layers.LayerNorm(8)),
            ('conv1', nn.Conv2d(8, 16, (1, 1), bias=False)),
            ('bias1', deepgaze_pytorch.layers.Bias(16)),
            ('softplus1', nn.Softplus()),

            ('layernorm2', deepgaze_pytorch.layers.LayerNorm(16)),
            ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
            ('bias2', deepgaze_pytorch.layers.Bias(1)),
            ('softplus2', nn.Softplus()),
        ]))
    
    def build_fixation_selection_network(self, scanpath_features=16):
        return nn.Sequential(OrderedDict([
            ('layernorm0', deepgaze_pytorch.layers.LayerNormMultiInput([1, scanpath_features])),
            ('conv0', deepgaze_pytorch.layers.Conv2dMultiInput([1, scanpath_features], 128, (1, 1), bias=False)),
            ('bias0', deepgaze_pytorch.layers.Bias(128)),
            ('softplus0', nn.Softplus()),

            ('layernorm1', deepgaze_pytorch.layers.LayerNorm(128)),
            ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
            ('bias1', deepgaze_pytorch.layers.Bias(16)),
            ('softplus1', nn.Softplus()),

            ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
        ]))


    def _make_network(self):
        if self.params.backbone_params is not None:
            class ToFloatImage(nn.Module):
                def forward(self, x):
                    x = x.float()
                    if x.max() > 1:
                        x = x / 255
                    sf = 224 / min(x.shape[2:])
                    # scale and resize image and center bias to match training regime
                    if sf != 1:
                        x = torch.nn.functional.interpolate(x, scale_factor=sf)
                    return x

            backbone = self.params.backbone_params.cls(self.params.backbone_params)
            backbone = nn.Sequential(
                ToFloatImage(),
                backbone
            )
            # ckp_path = self.params.backbone_config['ckp_path']
            # backbone = load_params_into_model(torch.load(ckp_path), backbone)
            feature_extractor = mFeatureExtractor(
                    backbone, self.params.backbone_config['feature_layers']
                )

        x = torch.rand(1,3,224,224)
        feats = feature_extractor(x)
        featdim = sum([f.shape[1] for f in feats])
        self.fixation_predictor = DeepGazeIIIModule(
            features=feature_extractor,
            saliency_network=self.build_saliency_network(featdim),
            scanpath_network=None,
            fixation_selection_network=self.build_fixation_selection_network(scanpath_features=0),
            downsample=1,
            readout_factor=4,
            saliency_map_factor=4,
            included_fixations=[],
        )

        if self.params.ckp_path is not None:
            ckp = {'state_dict': torch.load(self.params.ckp_path)}
            self.fixation_predictor = load_params_into_model(ckp,  self.fixation_predictor)
    
    def predict_fixation_map(self, image, centerbias):
        b,c,h,w = image.shape

        # get x and y from index
        x_hist = self.hist % w
        y_hist = self.hist // w
        
        fixation_map = self.fixation_predictor.forward(image, centerbias, None, None, None)
        fixation_map = fixation_map.unsqueeze(1)
        return fixation_map

    @classmethod
    def get_pretrained_model_params(cls, cfgstr):
        def set_param2(p:BaseParameters, value, param_name=None, param_type=None, default=None, set_all=False):
            # print(p)
            if (param_name is not None) and hasattr(p, param_name):
                setattr(p, param_name, value)
                if not set_all:
                    return True

            param_set = False
            if hasattr(p, 'asdict'):
                d = p.asdict(recurse=False)
                for k,v in d.items():            
                    if np.iterable(v):
                        for i,x in enumerate(v):
                            if isinstance(x, BaseParameters):
                                if (param_type is not None) and isinstance(x, param_type):
                                    v[i] = value
                                    param_set = p
                                else:
                                    param_set = set_param2(x, value, param_name, param_type, default, set_all)
                                if param_set and (not set_all):
                                    return param_set
                    else:
                        if (param_type is not None) and isinstance(v, param_type):
                            setattr(p, k, value)
                            param_set = True
                        else:
                            param_set = set_param2(v, value, param_name, param_type, default, set_all)
                        if param_set and (not set_all):
                            return param_set
            return param_set
        
        from neurips23.noisy_retina_blur import ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18
        from rblur.models import IdentityLayer
        from rblur.retina_preproc import AbstractRetinaFilter, GaussianNoiseLayer

        rblur_in1k_params = ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18().get_model_params()
        set_param2(rblur_in1k_params, IdentityLayer.get_params(), param_type=AbstractRetinaFilter.ModelParams)
        set_param2(rblur_in1k_params, IdentityLayer.get_params(), param_type=GaussianNoiseLayer.ModelParams)
        cfgdict = {
            'rblur-6.1-7.0-7.1-in1k':{
                                        'bbparams': rblur_in1k_params,
                                        'feature_layers':[
                                                            '1.classifier.resnet.6.1',
                                                            '1.classifier.resnet.7.0',
                                                            '1.classifier.resnet.7.1',
                                                        ],
                                        'ckp_path':'/home/mshah1/workhorse3/train_deepgaze3/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18/pretraining/final.pth',
                                        # 'ckp_path':'/ocean/projects/cis220031p/mshah1/train_deepgaze3/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18/pretraining/final.pth',
                                        'min_img_dim':224
                                    },
            'rwarp-6.1-7.0-7.1-in1k':{
                                        'bbparams': rblur_in1k_params,
                                        'feature_layers':[
                                                            '1.classifier.resnet.6.1',
                                                            '1.classifier.resnet.7.0',
                                                            '1.classifier.resnet.7.1',
                                                        ],
                                        'ckp_path':'/home/mshah1/workhorse3/train_deepgaze3/ImagenetRetinaWarpCyclicLRRandAugmentXResNet2x18/pretraining/step-0051.pth',
                                        'min_img_dim':224
                                    },
            'rblur-6.1-7.0-7.1-clickme-in1k':{
                                        'bbparams': rblur_in1k_params,
                                        'feature_layers':[
                                                            '1.classifier.resnet.6.1',
                                                            '1.classifier.resnet.7.0',
                                                            '1.classifier.resnet.7.1',
                                                        ],
                                        'ckp_path':'/home/mshah1/workhorse3/train_deepgaze3/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18/clickme/step-0002.pth',
                                        'min_img_dim':224
                                    },
            'rblur-4.1-6.1-7.0-7.1-in1k':{
                                        'bbparams': rblur_in1k_params,
                                        'feature_layers':[
                                                            '1.classifier.resnet.4.1',
                                                            '1.classifier.resnet.6.1',
                                                            '1.classifier.resnet.7.0',
                                                            '1.classifier.resnet.7.1',
                                                        ],
                                        'ckp_path':'/home/mshah1/workhorse3/train_deepgaze3/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18/layers_4_6_7/pretraining/final.pth',
                                        'min_img_dim':224
                                    },
            'rblur-3-4.1-5.1-7.0-7.1-in1k':{
                                        'bbparams': rblur_in1k_params,
                                        'feature_layers':[
                                                            '1.classifier.resnet.3',
                                                            '1.classifier.resnet.4.1',
                                                            '1.classifier.resnet.5.1',
                                                            '1.classifier.resnet.7.0',
                                                            '1.classifier.resnet.7.1',
                                                        ],
                                        'ckp_path':'/home/mshah1/workhorse3/train_deepgaze3/ImagenetNoisyRetinaBlurWRandomScalesCyclicLRRandAugmentXResNet2x18/layers_3_4_5_7/pretraining/final.pth',
                                        'min_img_dim':224
                                    },
        }
        cfg = cfgdict[cfgstr]
        p = cls.ModelParams(cls, cfg['bbparams'], 
                            {'feature_layers':cfg['feature_layers']},
                            ckp_path=cfg['ckp_path'],
                            min_image_dim=cfg['min_img_dim'])
        return p

class DeepGazeIIE(BaseFixationPredictor):
    def __init__(self, params: BaseFixationPredictor.ModelParams) -> None:
        super().__init__(params)
        self.centerbias_template = nn.parameter.Parameter(torch.FloatTensor(np.load('/home/mshah1/projects/adversarialML/biologically_inspired_models/DeepGaze/centerbias_mit1003.npy')), requires_grad=False)
    
    def _make_network(self):
        self.fixation_predictor = deepgaze_pytorch.DeepGazeIIE(self.params.pretrained)

    def predict_fixation_map(self, image, centerbias):
        fixation_map = self.fixation_predictor.forward(image, centerbias)
        return fixation_map

class DeepGazeII(BaseFixationPredictor):
    def __init__(self, params: BaseFixationPredictor.ModelParams) -> None:
        super().__init__(params)
        # self.centerbias_template = nn.parameter.Parameter(torch.FloatTensor(np.load('/home/mshah1/projects/adversarialML/biologically_inspired_models/DeepGaze/centerbias_mit1003.npy')), requires_grad=False)
    
    def _make_network(self):
        self.fixation_predictor = deepgaze_pytorch.DeepGazeI(self.params.pretrained)

    def predict_fixation_map(self, image, centerbias):
        fixation_map = self.fixation_predictor.forward(image, centerbias).unsqueeze(1)
        return fixation_map


class GaussianPyramid(nn.Module):
    def __init__(self, nlevels) -> None:
        super().__init__()
        self.nlevels = nlevels
        self.register_buffer('kernel', self._create_kernel())
    
    def _create_kernel(self):
        kernel = np.array((1., 4., 6., 4., 1.), dtype=np.float32)
        kernel = np.outer(kernel, kernel)
        kernel /= np.sum(kernel)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        return kernel
    
    def _downsample(self, x):
        ic = x.shape[1]
        if ic > 1:
            kernel = torch.repeat_interleave(self.kernel, ic, 0)
        else:
            kernel = self.kernel
        x = nn.ReflectionPad2d(kernel.shape[2]//2)(x)
        x = nn.functional.conv2d(x, kernel, stride=2, groups=ic)
        return x
    
    def forward(self, x):
        levels = [x]
        for _ in range(self.nlevels):
            x = self._downsample(x)
            levels.append(x)
        return levels
class FixationPredictionNetwork(BaseFixationPredictor):
    @define(slots=False)
    class ModelParams(BaseFixationPredictor.ModelParams):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        arch: str = 'unet'
        preprocessing_params: BaseParameters = None
        backbone_params: Union[BaseParameters, str, nn.Module] = 'resnet18'
        backbone_ckp_path: str = None
        freeze_backbone: bool = False
        partially_mask_loss: bool = False
        loss_fn: str = 'bce'
        llfeat_module_name: str = None
        llfeat_dim: int = None
        hlfeat_module_name: str = None
        hlfeat_dim: int = None

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
    def _make_network(self):
        x = torch.rand(1, *(self.params.common_params.input_size))
        if self.params.preprocessing_params is not None:
            self.preprocessor = self.params.preprocessing_params.cls(self.params.preprocessing_params)
        else:
            self.preprocessor = nn.Identity()
        # rn18 = xresnet18()
        # self.conv = nn.Sequential(*([rn18[i] for i in range(0, 8)]), nn.Conv2d(512, 1, 1))
        
        if self.params.arch == 'unet':
            self.conv = UNet(num_filters=[16,32,64,128,256])

        # self.conv = fcn_resnet50(pretrained=True)
        # self.conv.classifier[4] = nn.Conv2d(512, 1, 1, 1)
        # self.conv.aux_classifier[4] = nn.Conv2d(256, 1, 1, 1)

        # self.conv = deeplabv3_resnet50(True)
        # self.conv.classifier[4] = nn.Conv2d(256, 1, 1, 1)
        # self.conv = deeplabv3._deeplabv3_resnet(resnet18(False), 1, False)
        # self.conv.classifier = deeplabv3.DeepLabHead(512, 1)
        if self.params.arch == 'deepgaze2e':
            self.fixation_predictor = DeepGazeIIE(DeepGazeIIE.get_params())
            self.fixation_predictor.requires_grad_(False)
        if self.params.arch == 'deepgaze2':
            self.fixation_predictor = DeepGazeII(DeepGazeII.get_params())
            self.fixation_predictor.requires_grad_(False)
        if (self.params.arch == 'deeplab3p') or (self.params.arch == 'gala'):
            if self.params.backbone_params == 'resnet18':
                backbone = resnet18()
            elif isinstance(self.params.backbone_params, BaseParameters):
                backbone = self.params.backbone_params.cls(self.params.backbone_params)
            elif isinstance(self.params.backbone_params, nn.Module):
                backbone = self.params.backbone_params
            if self.params.backbone_ckp_path:
                ckp = torch.load(self.params.backbone_ckp_path)
                sd = ckp['state_dict']
                ckp['state_dict'] = {k[k.find('resnet'):]:v for k,v in sd.items() if 'resnet' in k}
                print(f'loading resnet backbone from {self.params.backbone_ckp_path}...')
                p1 = torch.nn.utils.parameters_to_vector(backbone.parameters())
                load_params_into_model(src_model=ckp, tgt_model=backbone)
                p2 = torch.nn.utils.parameters_to_vector(backbone.parameters())
                # print(torch.norm(p2-p1))
                print('backbone loaded!')
            if self.params.freeze_backbone:
                backbone.requires_grad_(False)
            if (self.params.llfeat_module_name and self.params.hlfeat_module_name and 
                    self.params.llfeat_dim and self.params.hlfeat_dim):
                if isinstance(backbone, XResNet34):
                    backbone = backbone.resnet
                backbone = IntermediateLayerGetter(backbone, {self.params.llfeat_module_name:'low_level_feats', 
                                                                self.params.hlfeat_module_name:'out'})
                if self.params.arch == 'deeplab3p':
                    self.fixation_predictor = DeepLab3p(backbone, 1, hl_channels=self.params.hlfeat_dim, ll_channels=self.params.llfeat_dim)
                if self.params.arch == 'gala':
                    self.fixation_predictor = GALA(backbone, self.params.hlfeat_dim)
            elif isinstance(backbone, ResNet):
                if self.params.arch == 'deeplab3p':
                    backbone = IntermediateLayerGetter(backbone, {'layer1':'low_level_feats', 'layer4':'out'})
                    self.fixation_predictor = DeepLab3p(backbone, 1)
                if self.params.arch == 'gala':
                    backbone = IntermediateLayerGetter(backbone, {'layer3':'out'})
                    self.fixation_predictor = GALA(backbone)
            elif isinstance(backbone, XResNet34):
                backbone = IntermediateLayerGetter(backbone.resnet, {'4':'low_level_feats', '7':'out'})
                if self.params.arch == 'deeplab3p':
                    self.fixation_predictor = DeepLab3p(backbone, 1, hl_channels=1024, ll_channels=128)
                if self.params.arch == 'gala':
                    self.fixation_predictor = GALA(backbone, 1024)
            else:
                raise ValueError(f'backbone must be a subclass of either torchvision.models.ResNet or rblur.models.XResNet34 but got {type(backbone)}')
        self.pyramid = GaussianPyramid(5)
        
    def preprocess(self, x):
        x = self.preprocessor(x)
        if isinstance(x, tuple):
            x = x[0]
        return x

    def forward(self, x):
        x = self.preprocess(x)
        x = self.fixation_predictor(x)#['out']
        # x = torch.flatten(x,1)
        # x = self.output(x)
        return x
    
    def reshape_target(self, y):
        if y.dim() == 2:
            n = int(np.sqrt(y.shape[-1]))
            y = y.reshape(y.shape[0], 1, n, n)
        return y

    def upsample_target(self, y, size):
        y = torch.nn.functional.interpolate(y, size=size)
        return y
    
    def standardize_heatmap(self, y):
        std, mean = torch.std_mean(y, (2,3), False, keepdim=True)
        y = (y-mean)/(std + 1e-5)
        return y

    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)

        y = self.reshape_target(y)
        y_us = self.upsample_target(y, logits.shape[2:])
        logit_levels = self.pyramid(logits)
        y_levels = self.pyramid(y_us)
        # print(y.shape, y_us.shape, [y_.shape for y_ in y_levels])
        for i,y_ in enumerate(y_levels):
            if y_.shape == y.shape:
                y_levels[i] = y
        loss = 0
        for yl, ylp in zip(y_levels, logit_levels):
            if self.params.loss_fn == 'bce':
                npos = yl.sum()
                poswt = (torch.numel(yl)-npos)/npos
                loss_ = nn.BCEWithLogitsLoss(pos_weight=poswt, reduction='none')(ylp, yl)
            elif self.params.loss_fn == 'mse':
                ylp = self.standardize_heatmap(ylp)
                yl = self.standardize_heatmap(yl)
                mask = 1 - ((yl < 0) & (ylp < 0)).float()
                loss_ = nn.MSELoss(reduction='none')(ylp, yl) * mask
            if self.params.partially_mask_loss:
                mask = (torch.rand_like(yl) <= np.random.uniform(1/float(torch.numel(yl)), 1)).float()
                loss_ = (loss_ * mask).sum()/mask.sum()
            else:
                loss_ = loss_.mean()
            loss += loss_
        loss /= len(y_levels)
        if return_logits:
            return logits[-1], loss
        else:
            return loss       
        # logits = self.forward(x)
        # npos = y.sum()
        # poswt = (torch.numel(y)-npos)/npos
        # # print(logits.dtype, logits.shape, y.dtype, y.shape)
        # loss = nn.BCEWithLogitsLoss(pos_weight=poswt)(logits, y)
        # if return_logits:
        #     return logits, loss
        # else:
        #     return loss

class DenseNet(nn.Module):
    def __init__(self, inchannels, outchannels, kernel_size, nlayers) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            convbnrelu(inchannels, outchannels, kernel_size, 1, kernel_size//2),
            *([convbnrelu(outchannels, outchannels, kernel_size, 1, kernel_size//2) for i in range(nlayers-1)])
        ])
    
    def forward(self, x):
        act_sum = 0
        prev_act = 0
        for i,l in enumerate(self.convs):
            if i > 0:
                prev_act = x
            x = l(x+act_sum)
            act_sum += prev_act
        return x

class FixationHeatmapPredictionNetwork(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        preprocessing_params: BaseParameters = None

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()

    def _make_network(self):
        self.preprocessor = self.params.preprocessing_params.cls(self.params.preprocessing_params)
        self.pyramid = GaussianPyramid(5)
        self.heatmap_predictor = DeepLab3p(resnet18(), 1)
        # self._create_convs()
        # self.conv = nn.Sequential(
        #     convbnrelu(3, 16, 7, 1, 3),
        #     convbnrelu(16, 32, 5, 1, 2),
        #     convbnrelu(32, 64, 5, 1, 2),
        #     # DenseNet(32, 64, 5, 3),
        #     # nn.Upsample(scale_factor=2),
        #     # convbnrelu(64, 32, 3, 1, 1),
        #     # convbnrelu(32, 16, 3, 1, 1),
        #     # convbnrelu(64, 64, 5, 2, 2),
        #     # DenseNet(64, 64, 5, 3),
        #     # convbnrelu(64, 256, 3, 1, 1),
        #     # nn.PixelShuffle(2),
        #     nn.Conv2d(64,1,1,1,0),
        #     # nn.PixelShuffle(2),
        #     # nn.ReLU()
        # )

    # def _create_convs(self):
    #     rn18 = xresnet18()
    #     self.downsample = nn.Sequential(*([rn18[i] for i in range(0, 8)]))
    #     self.upsample_layers = nn.ModuleList([
    #         nn.Sequential(ResBlock(1,512, 256*4), nn.PixelShuffle(2)),
    #         nn.Sequential(ResBlock(1,256, 128*4), nn.PixelShuffle(2)),
    #         nn.Sequential(ResBlock(1,128, 64*4), nn.PixelShuffle(2), ResBlock(1,64, 64), ResBlock(1,64, 64)),
    #         nn.Sequential(convbnrelu(64,64*4,3,1,1), nn.PixelShuffle(2)),
    #         nn.Sequential(convbnrelu(64,32,3,1,1), convbnrelu(32,32*4,3,1,1), nn.PixelShuffle(2)),
    #     ])
    #     self.output_layers = nn.ModuleList([
    #         nn.Conv2d(512, 1, 1),
    #         nn.Conv2d(256, 1, 1),
    #         nn.Conv2d(128, 1, 1),
    #         nn.Conv2d(64, 1, 1),
    #         nn.Conv2d(64, 1, 1),
    #         nn.Conv2d(32, 1, 1),
    #     ])

    def preprocess(self, x):
        x = self.preprocessor(x)
        if isinstance(x, tuple):
            x = x[0]
        return x

    def forward(self, x, return_all_levels=False):
        x = self.preprocess(x)
        return self.heatmap_predictor(x)
        # x = self.downsample(x)
        # outputs = [self.output_layers[0](x)]
        # for ul, ol in zip(self.upsample_layers, self.output_layers[1:]):
        #     x = ul(x)
        #     o = ol(x)
        #     outputs.append(o)
        # if return_all_levels:
        #     return outputs
        # return outputs[-1]
    
    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x, return_all_levels=True)
        logits = self.pyramid(logits)[::-1]
        y_levels = self.pyramid(y)[::-1]
        loss = 0
        for yl, ylp in zip(y_levels, logits):
            l1loss = torch.flatten(ylp.abs(), 1).mean()
            loss += nn.MSELoss()(ylp, yl) + 5e-4*l1loss
        # l1loss = torch.flatten(logits.abs(), 1).mean()
        # loss = nn.MSELoss()(logits, y) + 5e-4*l1loss
        if return_logits:
            return logits[-1], loss
        else:
            return loss

def _update_mask(fi, I, hks, mask, mask_kernel):
    fr, fc = (fi // mask.shape[3], fi % mask.shape[3])
    top, bot, left, right = fr-hks, fr+hks+1, fc-hks, fc+hks+1
    ktop, kleft = max(0, -top), max(0, -left)
    kbot = mask_kernel.shape[0] - max(0, bot - mask.shape[2])
    kright = mask_kernel.shape[1] - max(0, right - mask.shape[3])
    M = mask[I]
    M[..., max(0,top):bot, max(0,left):right] *= mask_kernel[ktop:kbot, kleft:kright]
    mask[I] = M
    return mask

def unnormalized_gkern(kernlen=256, std=128):
    def gaussian_fn(M, std):
        n = torch.cat([torch.arange(M//2,-1,-1), torch.arange(1, M//2+1)])
        # w = torch.exp(-0.5*((n/std)**2))/(np.sqrt(2*np.pi)*std)
        sig2 = 2 * std * std
        w = torch.exp(-n ** 2 / sig2)
        return w
    assert kernlen%2 == 1
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std) 
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d

class RetinaFilterWithFixationPrediction(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        preprocessing_params: BaseParameters = None
        retina_params: BaseParameters = None
        fixation_params: BaseParameters = None
        fixation_model_ckp: str = None
        freeze_fixation_model: bool = True
        target_downsample_factor: int = 32
        loc_sampling_temp: float = 5.
        num_train_fixation_points: int = 1
        num_eval_fixation_points: int = 1
        apply_retina_before_fixation: bool = True
        salience_map_provided_as_input_channel: bool = False
        random_fixation_prob: float = 0.
        disable: bool = False
        return_fixation_maps: bool = False
        return_fixated_images: bool = True
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()
        self.ckp_loaded = False
        if isinstance(self.retina, RBlur2):
            self.loc_offset = np.array(self.retina.input_shape[1:])//2
        self.interim_outputs = {}
        self.precomputed_fixation_map = None
    
    def _make_network(self):
        if self.params.preprocessing_params is not None:
            self.preprocessor = self.params.preprocessing_params.cls(self.params.preprocessing_params)
        else:
            self.preprocessor = nn.Identity()
        self.retina = self.params.retina_params.cls(self.params.retina_params)
        self.retina.loc_mode = self.retina.params.loc_mode = 'const'
        if hasattr(self.params.fixation_params, 'fixation_width_frac'):
            if hasattr(self.retina, '_get_view_scale'):
                s = self.retina._get_view_scale()
                w = 2*self.retina.clr_isobox_w[::-1][s] / self.params.target_downsample_factor
                frac = w / max(self.retina.input_shape[1:])
            else:
                frac = 0.5
            self.params.fixation_params.fixation_width_frac = frac
        self.fixation_model = self.params.fixation_params.cls(self.params.fixation_params)
        if self.params.freeze_fixation_model:
            self.fixation_model.requires_grad_(False)

    def _maybe_load_ckp(self):
        if (self.params.fixation_model_ckp is not None) and (not self.ckp_loaded):
            print('loading fixation model...')
            ckp = torch.load(self.params.fixation_model_ckp)
            load_params_into_model(ckp, self.fixation_model)
            print('fixation model loaded!')
            self.fixation_model = self.fixation_model.eval()
            self.ckp_loaded = True
    
    def set_precomputed_fixation_map(self, fmaps):
        self.precomputed_fixation_map = fmaps
    
    def unset_precomputed_fixation_map(self):
        self.precomputed_fixation_map = None
    
    def preprocess(self, x):
        x = self.preprocessor(x)
        if isinstance(x, tuple):
            x = x[0]
        return x

    def get_loc_from_fmap(self, x, fmap):
        if fmap.dim() > 1:
            if self.training:
                flat_fmap = fmap.reshape(-1)
                sample = nn.functional.gumbel_softmax(flat_fmap, tau=5)
                loc_importance_map = sample.reshape(fmap.shape)
            else:
                loc_importance_map = fmap
            loc = (loc_importance_map==torch.max(loc_importance_map)).nonzero()[0].cpu().detach().numpy()
            if isinstance(self.retina, RBlur2):
                loc = self.loc_offset - loc
                loc[loc < 0] = 0
        else:
            locidx = int(torch.argmax(fmap).cpu().detach())
            col_locs = np.linspace(0, x.shape[2]-1, int(np.sqrt(fmap.shape[0])), dtype=np.int32)
            row_locs = np.linspace(0, x.shape[1]-1, int(np.sqrt(fmap.shape[0])), dtype=np.int32)
            col = col_locs[locidx % len(col_locs)]
            row = row_locs[locidx // len(col_locs)]
            if isinstance(self.retina, RBlur2):
                loc = (max(0, self.loc_offset[0]-row), max(0, self.loc_offset[1]-col))
            else:
                loc = (row, col)
        return tuple(loc)
    
    def get_topk_fixations(self, fmap, K):
        n,_,h,w = fmap.shape
        std = max(h,w) / 10
        ks = int(4*std) 
        ks += int((ks % 2) == 0)
        hks = ks // 2
        mask = torch.ones((n,1,h,w), device=fmap.device)
        mask_kernel = 1-unnormalized_gkern(ks, std) + 1e-8
        mask_kernel = mask_kernel.to(fmap.device)
        # print(fmap.shape, mask.shape, self.mask_kernel.shape)
        
        if not self.params.salience_map_provided_as_input_channel:
            _update_mask(0, torch.arange(mask.shape[0]), hks, mask, mask_kernel)

        fxrows = []
        fxcols = []
        fxidxs = []
        for k in range(K):
            fmap = fmap + mask.log()
            if self.training:
                masked_flat_prob_map = gumbel_softmax(torch.flatten(fmap, 1), tau=self.params.loc_sampling_temp, hard=True)
                index_tensor = torch.arange(masked_flat_prob_map.shape[1], dtype=masked_flat_prob_map.dtype, device=masked_flat_prob_map.device).unsqueeze(1)
                fidx = masked_flat_prob_map.mm(index_tensor).squeeze(-1).int()
                
                rand_fidx = torch.randint_like(fidx, masked_flat_prob_map.shape[1])
                rand_idx = (torch.rand(fidx.shape[0]) < self.params.random_fixation_prob)
                fidx[rand_idx] = rand_fidx[rand_idx]
            else:
                fidx = torch.flatten(fmap, 1).argmax(1)
            # masked_prob_map = prob_map * mask
            # for i in range(n):
            #     plt.subplot(2*n, K, i*K + k+1)
            #     plt.imshow(convert_image_tensor_to_ndarray(masked_flat_prob_map.reshape(fmap.shape)[i]))
            #     plt.subplot(2*n, K, (i+1)*K + k+1)
            #     plt.imshow(convert_image_tensor_to_ndarray(fmap[i]))
            # plt.subplot(n, K, K + k+1)
            # plt.imshow(convert_image_tensor_to_ndarray(mask[0]), vmin=0., vmax=1.)                

            fxidxs.append(fidx)
            fxrows.append(fidx // fmap.shape[3])
            fxcols.append(fidx % fmap.shape[3])
            # print(k, fxrows[-1], fxcols[-1])
            if K > 1:
                fidx_set = set(fidx.cpu().detach().numpy().tolist())
                for fi in fidx_set:
                    _update_mask(fi, (fidx == fi), hks, mask, mask_kernel)
                # top, bot, left, right = fxrows[-1]-hks, fxrows[-1]+hks+1, fxcols[-1]-hks, fxcols[-1]+hks+1
                # ktop, kleft = torch.relu(-top), torch.relu(-left)
                # kbot = self.mask_kernel.shape[0] - torch.relu(bot - mask.shape[2])
                # kright = self.mask_kernel.shape[1] - torch.relu(right - mask.shape[3])
                # for i, (t, b, l, r, kt, kb, kl, kr) in enumerate(zip(top, bot, left, right, ktop, kbot, kleft, kright)):
                #     mask[i, 0, max(0,t):b, max(0,l):r] *= self.mask_kernel[kt:kb, kl:kr]
            
        fxrows = torch.stack(fxrows, 1)
        fxcols = torch.stack(fxcols, 1)
        fxidxs = torch.stack(fxidxs, 1)
        # print(fxrows[:3], fxcols[:3])
        return fxrows, fxcols, fxidxs
    
    def _maybe_downsample_fmap(self, fmaps):
        downsample_factor = self.params.target_downsample_factor
        if downsample_factor > 1:
            fmaps = nn.functional.avg_pool2d(fmaps, downsample_factor, stride=downsample_factor)
        return fmaps

    def get_loc_from_fmaps(self, fmaps):
        if fmaps.dim() > 2:
            downsample_factor = self.params.target_downsample_factor
            if downsample_factor > 1:
                fmaps = nn.functional.avg_pool2d(fmaps, downsample_factor, stride=downsample_factor)
            w = fmaps.shape[-1] # assumes that fmaps is square
            K = self.params.num_train_fixation_points if self.training else self.params.num_eval_fixation_points
            
            if (fmaps.dim() == 4):
                if fmaps.shape[1] == 1:
                    flat_fmaps = torch.flatten(fmaps, 1)
                    rows, cols, loc_idxs = self.get_topk_fixations(fmaps, K)
                else:
                    flat_fmaps = torch.flatten(fmaps, 2)
                    loc_idxs = torch.argmax(flat_fmaps, -1)
                    rows = loc_idxs // w
                    cols = loc_idxs % w
                loc_importance_maps = torch.log_softmax(flat_fmaps, -1)
                self.interim_outputs['loc_importance_maps'] = loc_importance_maps
                self.interim_outputs['selected_loc_idxs'] = loc_idxs
            else:
                ValueError(f'fixation maps must be 4D (batch x #fixations x height x width), but got {fmaps.dim()} tensor of shape {fmaps.shape}')
            
            if downsample_factor > 1:
                rows = rows * downsample_factor + downsample_factor//2
                cols = cols * downsample_factor + downsample_factor//2
            if isinstance(self.retina, RBlur2):
                rows = torch.relu(self.loc_offset - rows)
                cols = torch.relu(self.loc_offset - cols)
            locs = torch.stack([rows, cols], -1).detach().cpu()
            # locs = list(zip(rows, cols))
        return locs
    
    def get_fixations_from_model(self, x):
        x = self.preprocess(x)
        K = self.params.num_train_fixation_points if self.training else self.params.num_eval_fixation_points
        fxidxs = [torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)]
        fxrows = [torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)]
        fxcols = [torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)]
        fmaps = []
        x_out = torch.zeros_like(x).unsqueeze(1).repeat_interleave(K, 1)

        for k in range(K):
            if (k > 0):
                fmap = self.fixation_model(x_out[:, k-1], fidx)
            else:
                if self.params.apply_retina_before_fixation:
                    if isinstance(self.retina, RBlur2):
                        # self.retina.params.loc = tuple(self.loc_offset - np.array(x.shape[2:])//2)
                        self.retina.params.loc = (self.loc_offset, self.loc_offset)
                    else:
                        # self.retina.params.loc = tuple(np.array(x.shape[2:])//2)
                        self.retina.params.loc = (0,0)
                    x_blurred = self.retina(x)
                else:
                    x_blurred = x
                fmap = self.fixation_model(x_blurred)
            fmaps.append(fmap)

            fmap_ds = self._maybe_downsample_fmap(fmap)
            downsample_factor = self.params.target_downsample_factor
            
            if hasattr(self.fixation_model, 'sample_fixation'):
                fidx_ds = self.fixation_model.sample_fixation(fmap_ds)
            else:
                if self.training:
                    masked_flat_prob_map = gumbel_softmax(torch.flatten(fmap_ds, 1), tau=self.params.loc_sampling_temp, hard=True)
                else:
                    soft_masked_flat_prob_map = torch.softmax(torch.flatten(fmap_ds, 1), 1)
                    index = masked_flat_prob_map.max(1, keepdim=True)[1]
                    hard_masked_flat_prob_map = torch.zeros_like(masked_flat_prob_map, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
                    masked_flat_prob_map = hard_masked_flat_prob_map - soft_masked_flat_prob_map.detach() + soft_masked_flat_prob_map
                    # fidx_ds = masked_flat_prob_map.argmax(1)

                index_tensor = torch.arange(masked_flat_prob_map.shape[1], dtype=masked_flat_prob_map.dtype, device=masked_flat_prob_map.device).unsqueeze(1)
                fidx_ds = masked_flat_prob_map.mm(index_tensor).squeeze(-1).int()
            
            if self.training:
                rand_fidx = torch.randint_like(fidx_ds, torch.flatten(fmap_ds, 1).shape[1])
                rand_idx = (torch.rand(fidx_ds.shape[0]) < self.params.random_fixation_prob)
                fidx_ds[rand_idx] = rand_fidx[rand_idx]

            # if k == 0:
            #     plt.subplot(2, K+1, k+1)
            #     plt.imshow(convert_image_tensor_to_ndarray(x[0]))
            #     plt.plot([fxc[0].cpu().detach().numpy() for fxc in fxcols], [fxr[0].cpu().detach().numpy() for fxr in fxrows], 'o-', color='red')
            #     plt.axis('off')

            #     plt.subplot(2, K+1, k+2)
            #     plt.imshow(convert_image_tensor_to_ndarray(x_blurred[0]))
            # else:
            #     plt.subplot(2, K+1, k+2)
            #     plt.imshow(convert_image_tensor_to_ndarray(x_out[0, k-1]))
            # plt.plot([fxc[0].cpu().detach().numpy() for fxc in fxcols], [fxr[0].cpu().detach().numpy() for fxr in fxrows], 'o-', color='red')
            # plt.axis('off')

            # plt.subplot(2, K+1, (K+1) + k+2)
            # masked_flat_prob_map = torch.softmax(torch.flatten(fmap_ds, 1), 1)
            # # soft_masked_flat_prob_map = torch.softmax(torch.flatten(fmap_ds, 1), 1)
            # # index = soft_masked_flat_prob_map.max(1, keepdim=True)[1]
            # # hard_masked_flat_prob_map = torch.zeros_like(soft_masked_flat_prob_map, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
            # # masked_flat_prob_map = hard_masked_flat_prob_map - soft_masked_flat_prob_map.detach() + soft_masked_flat_prob_map
            # plt.imshow(convert_image_tensor_to_ndarray(masked_flat_prob_map[0].reshape(*(fmap_ds[0].shape))))
            # plt.axis('off')

            frow = (fidx_ds // fmap_ds.shape[3]) * downsample_factor + downsample_factor//2
            fcol = (fidx_ds % fmap_ds.shape[3]) * downsample_factor + downsample_factor//2
            fidx = frow * fmap.shape[3] + fcol

            fxidxs.append(fidx)
            fxcols.append(fcol)
            fxrows.append(frow)

            locs = fidx.cpu().detach().numpy()
            loc_set = set(fidx.cpu().detach().numpy().tolist())
            x_out_ = x_out[:, k]
            for loc in loc_set:
                r = loc // fmap.shape[3]
                c = loc % fmap.shape[3]
                # print(loc, (locs == loc))
                x_ = x[(locs == loc)]
                self.retina.params.loc = (r,c)
                x_ = self.retina(x_)
                x_out_[(locs == loc)] = x_
            x_out[:, k] = x_out_
        # plt.subplot(2, K+1, K+2)
        # plt.imshow(convert_image_tensor_to_ndarray(x_out[0].mean(0)))
        x_out = x_out.reshape(-1, *(x_out.shape[2:]))
        fmaps = torch.cat(fmaps, 1)
        return x_out, fmaps
        #     print(fidx_ds, frow, fcol, fidx, fidx//fmap.shape[3], fidx%fmap.shape[3])

            
        # fxidxs = torch.stack(fxidxs, 1)
        # fxcols = torch.stack(fxcols, 1)
        # fxrows = torch.stack(fxrows, 1)

        # if isinstance(self.retina, RBlur2):
        #     fxrows = torch.relu(self.loc_offset - fxrows)
        #     fxcols = torch.relu(self.loc_offset - fxcols)
        # locs = torch.stack([fxrows, fxcols], -1).detach().cpu()
        # return locs

    def forward(self, x, return_fixation_maps=False):
        if self.params.disable:
            return x
        if not self.ckp_loaded:
            self._maybe_load_ckp()
        self.retina.loc_mode = self.retina.params.loc_mode = 'const'
        if self.params.salience_map_provided_as_input_channel or (self.precomputed_fixation_map is not None):
            if self.params.salience_map_provided_as_input_channel:
                fixation_maps = x[:, 3:]
                # print('in fixation predictor', torch.norm(torch.flatten(fixation_maps,1), dim=1))
                x = x[:, :3]
            elif self.precomputed_fixation_map is not None:
                fixation_maps = self.precomputed_fixation_map
            x = self.preprocess(x)
            locs = self.get_loc_from_fmaps(fixation_maps)
            n_locs_per_img = locs.shape[1]
            locs = rearrange(locs, 'b n d -> (b n) d').numpy()
            loc_set = list(set([tuple(l) for l in locs]))
            loc_set = np.expand_dims(np.array(loc_set), 1)
            if n_locs_per_img > 1:
                x = torch.repeat_interleave(x, n_locs_per_img, 0)
            x_out = torch.zeros_like(x)
            for loc in loc_set:
                x_ = x[(locs == loc).all(1)]
                self.retina.params.loc = tuple(loc[0])
                x_ = self.retina(x_)
                x_out[(locs == loc).all(1)] = x_
        else:
            x_out, fixation_maps = self.get_fixations_from_model(x)
        
        output = None
        if self.params.return_fixated_images:
            output = (x_out, )
        if self.params.return_fixation_maps or return_fixation_maps:
            output = output + (fixation_maps,)
        if len(output) == 1:
            output = output[0]
        return output
        #     x = self.preprocess(x)
        #     if self.params.apply_retina_before_fixation:
        #         if isinstance(self.retina, RBlur2):
        #             # self.retina.params.loc = tuple(self.loc_offset - np.array(x.shape[2:])//2)
        #             self.retina.params.loc = (self.loc_offset, self.loc_offset)
        #         else:
        #             # self.retina.params.loc = tuple(np.array(x.shape[2:])//2)
        #             self.retina.params.loc = (0,0)
        #         x_blurred = self.retina(x)
        #     else:
        #         x_blurred = x
        #     fixation_maps = self.fixation_model(x_blurred)

        # locs = self.get_loc_from_fmaps(fixation_maps)
        n_locs_per_img = locs.shape[1]
        locs = rearrange(locs, 'b n d -> (b n) d').numpy()
        loc_set = list(set([tuple(l) for l in locs]))
        loc_set = np.expand_dims(np.array(loc_set), 1)
        
        if n_locs_per_img > 1:
            x = torch.repeat_interleave(x, n_locs_per_img, 0)
        print(locs)

        x_out = torch.zeros_like(x)
        for loc in loc_set:
            print(loc, (locs == loc).all(1))
            x_ = x[(locs == loc).all(1)]
            print(x_.shape)
            self.retina.params.loc = tuple(loc[0])
            x_ = self.retina(x_)
            x_out[(locs == loc).all(1)] = x_
            
        # out = []
        # for loc, x_ in zip(locs,x):
        #     # fmap = fmap.squeeze()
        #     # loc = self.get_loc_from_fmap(x_, fmap)
        #     self.retina.params.loc = loc
        #     # print(f'in RetinaFilterFixationPrediction, loc = {loc}')
        #     x_ = x_.unsqueeze(0)
        #     x_ = self.retina(x_)
        #     out.append(x_)
        # x_out = torch.cat(out, 0)
        
        
        # out = []
        # for loc, x_ in zip(locs,x):
        #     # fmap = fmap.squeeze()
        #     # loc = self.get_loc_from_fmap(x_, fmap)
        #     self.retina.params.loc = loc
        #     # print(f'in RetinaFilterFixationPrediction, loc = {loc}')
        #     x_ = x_.unsqueeze(0)
        #     x_ = self.retina(x_)
        #     out.append(x_)
        # x_out = torch.cat(out, 0)
        
        # def convert_image_tensor_to_ndarray(img):
        #     return img.cpu().detach().transpose(0,1).transpose(1,2).numpy()
        # nrows = n_locs_per_img
        # ncols = 2
        # for k in range(n_locs_per_img):
        #     plt.subplot(nrows,ncols,1+k*ncols)
        #     plt.imshow(convert_image_tensor_to_ndarray(x_out[k]))
        #     plt.subplot(nrows,ncols,2+k*ncols)
        #     fmap = fixation_maps[0]
        #     # fmap = torch.nn.functional.gumbel_softmax(fmap, tau=self.params.loc_sampling_temp, hard=False)
        #     plt.imshow(convert_image_tensor_to_ndarray(fmap))
        #     # loc = tuple(self.get_loc_from_fmap(x[0], fixation_maps[0].squeeze()))
        #     plt.title(f'fixation_point={locs[k]}')
        #     # plt.subplot(nrows,ncols,3+k*ncols)
        #     # self.retina.params.loc = loc
        #     # plt.imshow(convert_image_tensor_to_ndarray(self.retina(x[[k]])[0]))
        # plt.savefig('fixation_prediction/fixation_prediction_img.png')
        return x_out

    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = 0
        if return_logits:
            return logits, loss
        else:
            return loss

class MultiFixationTiedBackboneClassifier(RetinaFilterWithFixationPrediction):
    @define(slots=False)
    class ModelParams(RetinaFilterWithFixationPrediction.ModelParams):
        ensembling_mode: Literal['logit_mean', 'prob_sum'] = 'logit_mean'
        return_fixated_images: bool = False

    def get_fixations_from_model(self, x):
        x = self.preprocess(x)
        K = self.params.num_train_fixation_points if self.training else self.params.num_eval_fixation_points
        fxidxs = [torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)]
        fxrows = [torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)]
        fxcols = [torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)]
        fmaps = []
        x_out = torch.zeros_like(x).unsqueeze(1).repeat_interleave(K, 1)
        all_logits = []

        for k in range(K+1):
            if (k > 0):
                fmap = self.fixation_model(x_out[:, k-1], fidx)
            else:
                if self.params.apply_retina_before_fixation:
                    if isinstance(self.retina, RBlur2):
                        # self.retina.params.loc = tuple(self.loc_offset - np.array(x.shape[2:])//2)
                        self.retina.params.loc = (self.loc_offset, self.loc_offset)
                    else:
                        # self.retina.params.loc = tuple(np.array(x.shape[2:])//2)
                        self.retina.params.loc = (0,0)
                    x_blurred = self.retina(x)
                else:
                    x_blurred = x
                fmap = self.fixation_model(x_blurred)
            
            if isinstance(self.fixation_model, CustomBackboneDeepGazeIII) and isinstance(self.fixation_model.fixation_predictor.features, mFeatureExtractor):
                logits = self.fixation_model.fixation_predictor.features.final_output
                # logits = self.fixation_model.fixation_predictor.features(x_out[:, k-1], return_model_output=True)[1]
            else:
                raise NotImplementedError('fixation_model must be CustomBackboneDeepGazeIII')
            if k > 0:
                all_logits.append(logits)
            if k == K:
                break

            fmaps.append(fmap)

            fmap_ds = self._maybe_downsample_fmap(fmap)
            downsample_factor = self.params.target_downsample_factor
            
            if hasattr(self.fixation_model, 'sample_fixation'):
                fidx_ds = self.fixation_model.sample_fixation(fmap_ds)
            else:
                if self.training:
                    masked_flat_prob_map = gumbel_softmax(torch.flatten(fmap_ds, 1), tau=self.params.loc_sampling_temp, hard=True)
                else:
                    soft_masked_flat_prob_map = torch.softmax(torch.flatten(fmap_ds, 1), 1)
                    index = masked_flat_prob_map.max(1, keepdim=True)[1]
                    hard_masked_flat_prob_map = torch.zeros_like(masked_flat_prob_map, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
                    masked_flat_prob_map = hard_masked_flat_prob_map - soft_masked_flat_prob_map.detach() + soft_masked_flat_prob_map

                index_tensor = torch.arange(masked_flat_prob_map.shape[1], dtype=masked_flat_prob_map.dtype, device=masked_flat_prob_map.device).unsqueeze(1)
                fidx_ds = masked_flat_prob_map.mm(index_tensor).squeeze(-1).int()
            
            if self.training:
                rand_fidx = torch.randint_like(fidx_ds, torch.flatten(fmap_ds, 1).shape[1])
                rand_idx = (torch.rand(fidx_ds.shape[0]) < self.params.random_fixation_prob)
                fidx_ds[rand_idx] = rand_fidx[rand_idx]

            frow = (fidx_ds // fmap_ds.shape[3]) * downsample_factor + downsample_factor//2
            fcol = (fidx_ds % fmap_ds.shape[3]) * downsample_factor + downsample_factor//2
            fidx = frow * fmap.shape[3] + fcol

            fxidxs.append(fidx)
            fxcols.append(fcol)
            fxrows.append(frow)

            locs = fidx.cpu().detach().numpy()
            loc_set = set(fidx.cpu().detach().numpy().tolist())
            x_out_ = x_out[:, k]
            for loc in loc_set:
                r = loc // fmap.shape[3]
                c = loc % fmap.shape[3]
                # print(loc, (locs == loc))
                x_ = x[(locs == loc)]
                self.retina.params.loc = (r,c)
                x_ = self.retina(x_)
                x_out_[(locs == loc)] = x_
            x_out[:, k] = x_out_
        x_out = x_out.reshape(-1, *(x_out.shape[2:]))
        fmaps = torch.cat(fmaps, 1)
        all_logits = torch.stack(all_logits, 1)
        if self.params.ensembling_mode == 'logit_mean':
            all_logits = all_logits.mean(1)
        if self.params.ensembling_mode == 'prob_sum':
            all_logits = torch.log_softmax(all_logits, -1).logsumexp(1)
        return x_out, fmaps, all_logits
    
    def forward(self, x, return_fixation_maps=False):
        if self.params.disable:
            return x
        if not self.ckp_loaded:
            self._maybe_load_ckp()
        self.retina.loc_mode = self.retina.params.loc_mode = 'const'
        x_out, fixation_maps, logits = self.get_fixations_from_model(x)
        output = (logits,)
        if self.params.return_fixated_images:
            output = output + (x_out,)
        if self.params.return_fixation_maps or return_fixation_maps:
            output = output + (fixation_maps,)
        if len(output) == 1:
            output = output[0]
        return output
    
    def compute_loss(self, x, y, return_logits=True, **kwargs):
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        output = (loss,)
        if return_logits:
            output = (logits,) + output
        return output

class TiedBackboneRetinaFixationPreditionClassifier(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        retina_filter_params: RetinaFilterWithFixationPrediction.ModelParams = None
        backbone_params: BaseParameters = None

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()
        self.forward(torch.rand(2, *(self.params.common_params.input_size)))

    def _make_network(self):
        self.backbone = self.params.backbone_params.cls(self.params.backbone_params)
        self.params.retina_filter_params.fixation_params.backbone_params = self.backbone
        self.retina_filter = self.params.retina_filter_params.cls(self.params.retina_filter_params)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.params.common_params.dropout_p),
            nn.LazyLinear(self.params.common_params.num_units)
        )
    
    def preprocess(self, x):
        return self.retina_filter(x)

    def forward(self, x):
        x_ret = self.retina_filter(x)
        feat = self.backbone(x_ret)
        logits = self.classifier(feat)
        return logits
    
    def compute_loss(self, x, y, fixation_logits=None, return_logits=True, **kwargs):
        loss = 0
        if fixation_logits is not None:
            x_ret, loss = self.retina_filter.compute_loss(x, fixation_logits, return_logits=True)
            feat = self.backbone(x_ret)
            logits = self.classifier(feat)
        else:
            logits = self.forward(x)
        if self.training:
            num_fixation_points = self.params.retina_filter_params.num_train_fixation_points
        else:
            num_fixation_points = self.params.retina_filter_params.num_eval_fixation_points
        y = torch.repeat_interleave(y, num_fixation_points, 0)
        loss += nn.functional.cross_entropy(logits, y)
        loc_importance_maps = self.retina_filter.interim_outputs.get('loc_importance_maps', None)
        selected_locs = self.retina_filter.interim_outputs.get('selected_loc_idxs', None)
        if (loc_importance_maps is not None) and (selected_locs is not None):
            loc_importance_maps = torch.repeat_interleave(loc_importance_maps, num_fixation_points, 0)
            selected_locs = selected_locs.reshape(-1)
            selected_loc_imp = loc_importance_maps[torch.arange(loc_importance_maps.shape[0]), selected_locs]
            # is_correct = (torch.argmax(logits, 1) == y).float()
            # fixation_loss = -(is_correct*selected_loc_imp + (1-is_correct)*log1mexp(selected_loc_imp)).mean()
            is_correct = (torch.argmax(logits, 1) == y)
            fixation_loss = -(selected_loc_imp[is_correct].sum() + log1mexp(selected_loc_imp[~is_correct]).sum())/selected_loc_imp.shape[0]
            loss += fixation_loss

        # if self.training:
        #     print(logits[:2], loss)
        output = (loss,)
        if return_logits:
            output = (logits,) + output
        return output