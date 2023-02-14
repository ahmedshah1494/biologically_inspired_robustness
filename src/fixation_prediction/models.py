
from attrs import define, field
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
import numpy as np

from torch import nn
import torch
from fastai.vision.models.xresnet import xresnet18
from fastai.layers import ResBlock
from fastai.layers import ResBlock
from adversarialML.biologically_inspired_models.src.runners import load_params_into_model
from torchvision.models.segmentation import fcn_resnet50, fcn, deeplabv3_resnet50, deeplabv3
from torchvision.models.resnet import resnet18
from torchvision.models._utils import IntermediateLayerGetter
from matplotlib import pyplot as plt

from adversarialML.biologically_inspired_models.src.models import CommonModelParams, ConvEncoder, convbnrelu, bnrelu

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

class DeepLayer3p(nn.Module):
    def __init__(self, backbone, nclasses, upsample_output=True) -> None:
        super().__init__()
        self.backbone = IntermediateLayerGetter(backbone, {'layer1':'low_level_feats', 'layer4':'out'})
        self.aspp_layer = deeplabv3.ASPP(512, [1, 6, 12, 18], 256)
        self.decoder_conv1 = convbnrelu(64, 64, 1, 1, 0)
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
class FixationPredictionNetwork(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        arch: str = 'unet'
        preprocessing_params: BaseParameters = None

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()

    def _make_network(self):
        x = torch.rand(1, *(self.params.common_params.input_size))
        self.preprocessor = self.params.preprocessing_params.cls(self.params.preprocessing_params)
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

        # self.conv = DeepLayer3p(resnet18(), 1)
        self.pyramid = GaussianPyramid(5)
        
    def preprocess(self, x):
        x = self.preprocessor(x)
        if isinstance(x, tuple):
            x = x[0]
        return x

    def forward(self, x):
        x = self.preprocess(x)
        x = self.conv(x)#['out']
        # x = torch.flatten(x,1)
        # x = self.output(x)
        return x
    
    def reshape_target(self, y):
        n = int(np.sqrt(y.shape[-1]))
        y = y.reshape(y.shape[0], 1, n, n)
        return y

    def upsample_target(self, y, size):
        y = torch.nn.functional.interpolate(y, size=size)
        return y

    def compute_loss(self, x, y, return_logits=True):
        y = self.reshape_target(y)
        logits = self.forward(x)
        y_us = self.upsample_target(y, logits.shape[2:])
        logit_levels = self.pyramid(logits)
        y_levels = self.pyramid(y_us)
        # print(y.shape, y_us.shape, [y_.shape for y_ in y_levels])
        assert y_levels[-1].shape == y.shape
        y_levels[-1] = y
        loss = 0
        for yl, ylp in zip(y_levels, logit_levels):
            npos = yl.sum()
            poswt = (torch.numel(yl)-npos)/npos
            # print(logits.dtype, logits.shape, y.dtype, y.shape)
            loss += nn.BCEWithLogitsLoss(pos_weight=poswt)(ylp, yl)
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
        self.heatmap_predictor = DeepLayer3p(resnet18(), 1)
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

class RetinaFilterFixationPrediction(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        preprocessing_params: BaseParameters = None
        retina_params: BaseParameters = None
        fixation_params: BaseParameters = None
        fixation_model_ckp: str = None
    
    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()
        self.ckp_loaded = False
        self.loc_offset = np.array(self.retina.input_shape[1:])//2
    
    def _make_network(self):
        self.preprocessor = self.params.preprocessing_params.cls(self.params.preprocessing_params)
        self.retina = self.params.retina_params.cls(self.params.retina_params)
        self.retina.loc_mode = 'const'
        self.fixation_model = self.params.fixation_params.cls(self.params.fixation_params)

    def _maybe_load_ckp(self):
        if (self.params.fixation_model_ckp is not None) and (not self.ckp_loaded):
            print('loading fixation model...')
            ckp = torch.load(self.params.fixation_model_ckp)
            load_params_into_model(ckp, self.fixation_model)
            print('fixation model loaded!')
            self.fixation_model = self.fixation_model.eval()
            self.ckp_loaded = True
    
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
                fmap = sample.reshape(fmap.shape)
            loc = (fmap==torch.max(fmap)).nonzero()[0].cpu().detach().numpy()
            loc = self.loc_offset - loc
            loc[loc < 0] = 0
        else:
            locidx = int(torch.argmax(fmap).cpu().detach())
            col_locs = np.linspace(0, x.shape[2]-1, int(np.sqrt(fmap.shape[0])), dtype=np.int32)
            row_locs = np.linspace(0, x.shape[1]-1, int(np.sqrt(fmap.shape[0])), dtype=np.int32)
            col = col_locs[locidx % len(col_locs)]
            row = row_locs[locidx // len(col_locs)]
            loc = (max(0, self.loc_offset[0]-row), max(0, self.loc_offset[1]-col))
        return loc


    def forward(self, x):
        if not self.ckp_loaded:
            self._maybe_load_ckp()
        x = self.preprocess(x)
        self.retina.params.loc = tuple(self.loc_offset - np.array(x.shape[2:])//2)
        x = self.retina(x)
        fixation_maps = self.fixation_model(x)
        out = []
        for fmap, x_ in zip(fixation_maps,x):
            fmap = fmap.squeeze()
            # if fmap.dim() > 1:
            #     loc = (fmap==torch.max(fmap)).nonzero()[0].cpu().detach().numpy()
            #     loc = self.loc_offset - loc
            #     loc[loc < 0] = 0
            loc = self.get_loc_from_fmap(x_, fmap)
            self.retina.params.loc = loc
            x_ = x_.unsqueeze(0)
            x_ = self.preprocess(x_)
            x_ = self.retina(x_)
            out.append(x_)
        x = torch.cat(out, 0)
        return x

    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = 0
        if return_logits:
            return logits, loss
        else:
            return loss