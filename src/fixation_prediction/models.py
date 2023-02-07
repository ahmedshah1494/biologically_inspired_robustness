
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
from matplotlib import pyplot as plt

from adversarialML.biologically_inspired_models.src.models import CommonModelParams, ConvEncoder, convbnrelu

class FixationPredictionNetwork(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        conv_params: ConvEncoder.ModelParams = field(factory=ConvEncoder.ModelParams)
        output_layer_params: BaseParameters = None
        num_fixation_points: int = 49
        preprocessing_params: BaseParameters = None

    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self._make_network()

    def _make_network(self):
        x = torch.rand(1, *(self.params.common_params.input_size))
        self.preprocessor = self.params.preprocessing_params.cls(self.params.preprocessing_params)
        
        # x = self.preprocess(x)
        # self.params.conv_params.common_params.input_size = x.shape[1:]
        # self.conv = self.params.conv_params.cls(self.params.conv_params)

        # self.conv = nn.Sequential(
        #     convbnrelu(3, 16, 7, 2, 3),
        #     convbnrelu(16, 32, 5, 2, 2),
        #     convbnrelu(32, 64, 5, 2, 2),
        #     convbnrelu(64, 128, 5, 2, 2),
        #     convbnrelu(128, 256, 5, 2, 2),
        #     nn.Conv2d(256, 1, 1, 1)
        # )
        rn18 = xresnet18()
        self.conv = nn.Sequential(*([rn18[i] for i in range(0, 8)]), nn.Conv2d(512, 1, 1))

        # x = self.conv(x)
        # x = x.reshape(x.shape[0], -1)
        # self.output = self.params.output_layer_params.cls(self.params.output_layer_params)
        
    def preprocess(self, x):
        x = self.preprocessor(x)
        if isinstance(x, tuple):
            x = x[0]
        return x

    def forward(self, x):
        x = self.preprocess(x)
        x = self.conv(x)
        x = torch.flatten(x,1)
        # x = self.output(x)
        return x
    
    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        npos = y.sum()
        poswt = (torch.numel(y)-npos)/npos
        # print(logits.dtype, logits.shape, y.dtype, y.shape)
        loss = nn.BCEWithLogitsLoss(pos_weight=poswt)(logits, y)
        if return_logits:
            return logits, loss
        else:
            return loss

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
        self._create_convs()
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

    def _create_convs(self):
        rn18 = xresnet18()
        self.downsample = nn.Sequential(*([rn18[i] for i in range(0, 8)]))
        self.upsample_layers = nn.ModuleList([
            nn.Sequential(ResBlock(1,512, 256*4), nn.PixelShuffle(2)),
            nn.Sequential(ResBlock(1,256, 128*4), nn.PixelShuffle(2)),
            nn.Sequential(ResBlock(1,128, 64*4), nn.PixelShuffle(2), ResBlock(1,64, 64), ResBlock(1,64, 64)),
            nn.Sequential(convbnrelu(64,64*4,3,1,1), nn.PixelShuffle(2)),
            nn.Sequential(convbnrelu(64,32,3,1,1), convbnrelu(32,32*4,3,1,1), nn.PixelShuffle(2)),
        ])
        self.output_layers = nn.ModuleList([
            nn.Conv2d(512, 1, 1),
            nn.Conv2d(256, 1, 1),
            nn.Conv2d(128, 1, 1),
            nn.Conv2d(64, 1, 1),
            nn.Conv2d(64, 1, 1),
            nn.Conv2d(32, 1, 1),
        ])

    def preprocess(self, x):
        x = self.preprocessor(x)
        if isinstance(x, tuple):
            x = x[0]
        return x

    def forward(self, x, return_all_levels=False):
        x = self.preprocess(x)
        x = self.downsample(x)
        outputs = [self.output_layers[0](x)]
        for ul, ol in zip(self.upsample_layers, self.output_layers[1:]):
            x = ul(x)
            o = ol(x)
            outputs.append(o)
        if return_all_levels:
            return outputs
        return outputs[-1]
    
    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x, return_all_levels=True)
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
            load_params_into_model(torch.load(self.params.fixation_model_ckp), self.fixation_model)
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