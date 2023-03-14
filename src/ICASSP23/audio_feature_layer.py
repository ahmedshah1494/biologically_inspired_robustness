from turtle import forward
from typing import Optional, Tuple
from attrs import define
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from torchaudio.transforms import MFCC, Resample, MelSpectrogram, FrequencyMasking, TimeMasking, TimeStretch
import numpy as np
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

class MelSpectrogramExtractionLayer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        sample_rate: int = 16000
        n_fft: int = 400
        win_length: Optional[int] = None
        hop_length: Optional[int] = None
        f_min: float = 0.0
        f_max: Optional[float] = None
        pad: int = 0
        n_mels: int = 128
        power: float = 2.0
        normalized: bool = False
        center: bool = True
        pad_mode: str = 'reflect'
        onesided: bool = True
        norm: Optional[str] = None
        mel_scale: str = 'htk'
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.melspec = MelSpectrogram(**(params.asdict(filter=lambda a,v: a.name != 'cls')))
    
    def forward(self, x):
        return self.melspec(x)

    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = torch.zeros((x.shape[0],), device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

class LogMelSpectrogramExtractionLayer(MelSpectrogramExtractionLayer):
    def forward(self,x):
        melspec = super().forward(x)+1e-5
        logmelspec = torch.log(melspec)
        return logmelspec

class SpecAugmentLayer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        freq_mask_param: int = None
        time_mask_param: int = None
        time_stretch_range: Tuple[float, float] = None
        hop_length: int = None
        n_freq: int = 201,
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self.freq_mask = FrequencyMasking(self.params.freq_mask_param)
        self.time_mask = TimeMasking(self.params.time_mask_param)
        if self.params.time_stretch_range:
            self.time_stretch = TimeStretch(self.params.hop_length, self.params.n_freq)
    
    def forward(self, x):
        if self.training:
            x = self.freq_mask(x)
            x = self.time_mask(x)
            if self.params.time_stretch_range:
                x = self.time_stretch(x, np.random.uniform(*(self.params.time_stretch_range)))
        return x
    
    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = torch.zeros((x.shape[0],), device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss