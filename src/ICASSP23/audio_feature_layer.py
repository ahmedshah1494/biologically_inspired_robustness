from turtle import forward
from attrs import define
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from torchaudio.transforms import MFCC, Resample
import torch

class ResamplingLayer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        orig_freq: int = 16000
        new_freq: int = 16000
        resampling_method: str = 'sinc_interpolation'
        lowpass_filter_width: int = 6
        rolloff: float = 0.99
        beta: float = None,
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.resample = Resample(params.orig_freq, params.new_freq, params.resampling_method, 
        params.lowpass_filter_width, params.rolloff, params.beta)
    
    def __repr__(self):
        return f'ResamplingLayer(orig_freq={self.params.orig_freq}, new_freq={self.params.new_freq})'

    def forward(self, x):
        x = self.resample(x)
        return x
    
    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = torch.zeros((x.shape[0],), device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

class MFCCExtractionLayer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        sample_rate: int = 16000
        n_mfcc: int = 40
        dct_type: int = 2
        norm: str = 'ortho'
        log_mels: bool = False
        melkwargs: dict = None
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.mfcc = MFCC(params.sample_rate, params.n_mfcc, params.dct_type,
                            params.norm, params.log_mels, params.melkwargs)

    def forward(self, x):
        x = self.mfcc(x)
        return x
    
    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = torch.zeros((x.shape[0],), device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss