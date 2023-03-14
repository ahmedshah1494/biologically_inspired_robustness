from typing import List, Tuple
from attrs import define, field
import numpy as np
import torch
from torch import nn
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from adversarialML.biologically_inspired_models.src.models import CommonModelMixin, CommonModelParams, IdentityLayer, ConsistentActivationLayer
# from torchaudio.models.decoder import ctc_decoder
import sentencepiece as spm
from ctcdecode import CTCBeamDecoder
from multiprocessing import cpu_count
from einops import rearrange

@define(slots=False)
class ConvParams:
    out_channels: int = 1
    kernel_size: int = 1
    stride: int = 1
    padding: int = 0
    dilation: int = 1

class Conv1dEncoder(AbstractModel, CommonModelMixin):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        conv_params: List[ConvParams] = []
        group_norm: bool = False
        transpose_chan_and_time_dims: bool = False

    def __init__(self, params: ModelParams) -> None:
        super(Conv1dEncoder, self).__init__(params)
        self.params = params
        self.load_common_params()
        self._make_name()
        self._make_network()
    
    def _make_network(self):
        layers = []
        prev_channels = self.input_size[0]
        conv_params = self.params.conv_params
        for cp in conv_params:
            layers.append(nn.Conv1d(prev_channels, cp.out_channels, cp.kernel_size, cp.stride, cp.padding, cp.dilation, bias=self.use_bias))
            prev_channels = cp.out_channels
            if self.params.group_norm:
                layers.append(nn.GroupNorm(1, cp.out_channels))
            layers.append(self.activation())
            if self.dropout_p > 0:
                layers.append(nn.Dropout2d(self.dropout_p))
        self.conv_model = nn.Sequential(*layers)
    
    def forward(self, x, return_state_hist=False, **kwargs):
        if self.params.transpose_chan_and_time_dims:
            x = x.transpose(1,2)
        feat = self.conv_model(x)
        if self.params.transpose_chan_and_time_dims:
            feat = feat.transpose(1,2)
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

class ScanningConsistentActivationLayer1d(ConsistentActivationLayer):
    @define(slots=False)
    class ModelParams(ConsistentActivationLayer.ModelParams):
        conv_params: ConvParams = field(factory=ConvParams)
        transpose_chan_and_time_dims: bool = False

    def __init__(self, params: ModelParams) -> None:
        params.common_params.input_size = list(params.common_params.input_size)
        params.common_params.input_size[1] = params.conv_params.kernel_size
        params.common_params.num_units = params.conv_params.out_channels
        super().__init__(params)
        self.unfold = nn.Unfold((1,params.conv_params.kernel_size), (1,params.conv_params.dilation), 
                                (0,params.conv_params.padding), (1,params.conv_params.stride))
    def forward(self, x, return_state_hist=False, loop=False):
        if self.params.transpose_chan_and_time_dims:
            x = x.transpose(1,2)
        b, c, t = x.shape
        x = x.unsqueeze(2)
        x = self.unfold(x)
        if loop:
            x = rearrange(x, 'b d l -> l b d')
            new_x = []
            for x_ in x:
                outputs = super().forward(x_, return_state_hist)
                if isinstance(outputs, tuple):
                    x_ = outputs[0]
                else:
                    x_ = outputs
                new_x.append(x_)
            x = torch.stack(new_x, 2)
        else:
            x = rearrange(x, 'b d l -> (b l) d')
            outputs = super().forward(x, return_state_hist)
            if isinstance(outputs, tuple):
                x = outputs[0]
                sh = outputs[1]
            else:
                x = outputs
            x = rearrange(x, '(b l) d -> b d l', b=b)
        if self.params.transpose_chan_and_time_dims:
            x = x.transpose(1,2)
        return x
    
    def compute_loss(self, x, y, return_state_hist=False, return_logits=False):
        logits = self.forward(x)
        loss = torch.tensor(0., device=x.device)
        output = (loss,)
        if return_state_hist:
            output = output + (None,)
        if return_logits:
            output = (logits,) + output
        return output
    

class TDNNCTCASRModel(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        encoder_params: BaseParameters = None
        classifier_params: BaseParameters = None
        preprocessor_params:BaseParameters = None
        sentencepiece_model_path: str = None
        kenlm_model_path: str = None
        decoding_alpha: float = 0.
        decoding_beta: float = 0.
        decoding_beam_width: int = 20
    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        self.sentencepiece_model = spm.SentencePieceProcessor(model_file=self.params.sentencepiece_model_path)
        self.vocab = [self.sentencepiece_model.IdToPiece(i) for i in range(len(self.sentencepiece_model))]
        self._make_network()
    
    def _make_network(self):
        self.encoder = self.params.encoder_params.cls(self.params.encoder_params)
        self.classifier = self.params.classifier_params.cls(self.params.classifier_params)
        self.preprocessor = self.params.preprocessor_params.cls(self.params.preprocessor_params)
        self.ctc_decoder = CTCBeamDecoder(self.vocab, beam_width=self.params.decoding_beam_width, num_processes=cpu_count(), 
                                        model_path=self.params.kenlm_model_path, alpha=self.params.decoding_alpha, beta=self.params.decoding_beta)
        # self.decoder = ctc_decoder(tokens=self.vocab, blank_token='<blank>')
        # self.decoder = GreedyCTCDecoder(self.vocab)

    def preprocess(self, x):
        x = self.preprocessor(x)
        if isinstance(x, tuple):
            x = x[0]
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.preprocess(x)
        feat = self.encoder(x)
        if isinstance(feat, tuple):
            feat = feat[0]
        feat = feat.transpose(1,2)
        logits = self.classifier(feat)
        return logits
    
    def decode(self, x, greedy=False, return_idxs=False):
        logits = self.forward(x)
        if greedy:
            indices = torch.argmax(logits, dim=-1)  # [num_seq,]
            indices = torch.unique_consecutive(indices, dim=-1).cpu().detach().numpy().tolist()
            decoded_idxs = []
            decode = []
            for idxs in indices:
                idxs = [i for i in idxs if i != 0]
                decoded_idxs.append(idxs)
                d = self.sentencepiece_model.DecodeIds(idxs)
                decode.append(d)            
        else:
            probs = torch.softmax(logits, 2)
            beam_results, beam_scores, timesteps, out_lens = self.ctc_decoder.decode(probs)
            decoded_idxs = [d[:l] for d,l in zip(beam_results[:, 0], out_lens[:, 0])]
            decode = [self.sentencepiece_model.DecodeIds(d.cpu().detach().numpy().tolist()) for d in decoded_idxs]
        if return_idxs:
            return decode, decoded_idxs
        else:
            return decode

    def _compute_ctc_loss(self, logits, x, y, xlens, ylens):
        # print('torch.isfinite(logits) =',torch.isfinite(logits).all())
        logprobs = torch.log_softmax(logits.transpose(0,1), 2)
        # print('torch.isfinite(logprobs) =',torch.isfinite(logprobs).all())
        dsfactor = logprobs.shape[0] / xlens.max()
        xlens = torch.clamp(torch.ceil(xlens * dsfactor), 0, logprobs.shape[0]).long()
        loss = nn.CTCLoss()(logprobs, y, xlens, ylens)
        return loss

    def compute_loss(self, x, y, xlens, ylens, return_logits=True):
        logits = self.forward(x)
        loss = self._compute_ctc_loss(logits, x, y, xlens, ylens)
        if return_logits:
            return logits, loss
        return loss
