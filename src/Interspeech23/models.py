from typing import List
from attrs import define, field
import torch
from torch import nn
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from adversarialML.biologically_inspired_models.src.models import CommonModelMixin, CommonModelParams, ConvParams, IdentityLayer, ScanningConsistentActivationLayer
# from torchaudio.models.decoder import ctc_decoder
import sentencepiece as spm
from ctcdecode import CTCBeamDecoder
from multiprocessing import cpu_count

@define(slots=False)
class ConvParams(ConvParams):
    dilation: List[int] = None

class Conv1dEncoder(AbstractModel, CommonModelMixin):
    @define(slots=False)
    class ModelParams(BaseParameters):
        common_params: CommonModelParams = field(factory=CommonModelParams)
        conv_params: ConvParams = field(factory=ConvParams)

    def __init__(self, params: ModelParams) -> None:
        super(Conv1dEncoder, self).__init__(params)
        self.params = params
        self.load_common_params()
        self._load_conv_params()
        self._make_name()
        self._make_network()

    def _load_conv_params(self):
        self.kernel_sizes = self.params.conv_params.kernel_sizes
        self.strides = self.params.conv_params.strides
        self.padding = self.params.conv_params.padding
        self.dilation = self.params.conv_params.dilation
    
    def _make_network(self):
        layers = []
        nfilters = [self.input_size[0], *self.num_units]
        kernel_sizes = [None] + self.kernel_sizes
        strides = [None] + self.strides
        padding = [None] + self.padding
        dilation = [None] + self.dilation
        for i, (k,s,f,p,d) in enumerate(zip(kernel_sizes, strides, nfilters, padding, dilation)):
            if i > 0:
                layers.append(nn.Conv1d(nfilters[i-1], f, k, s, p, d, bias=self.use_bias))
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

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward_single(self, emission: torch.Tensor):
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()
    
    def forward(self, emissions):
        return [self.forward_single(e) for e in emissions]

class TDNNCTCASRModel(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        encoder_params: BaseParameters = None
        classifier_params: BaseParameters = None
        preprocessor_params:BaseParameters = None
        sentencepiece_model_path: str = None
    
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
        feat = feat.transpose(1,2)
        logits = self.classifier(feat)
        return logits
    
    def decode(self, x, greedy=False):
        logits = self.forward(x)
        if greedy:
            indices = torch.argmax(logits, dim=-1)  # [num_seq,]
            indices = torch.unique_consecutive(indices, dim=-1).cpu().detach().numpy().tolist()
            decode = []
            for idxs in indices:
                idxs = [i for i in idxs if i != 0]
                d = self.sentencepiece_model.DecodeIds(idxs)
                decode.append(d)            
        else:
            decoder = CTCBeamDecoder(self.vocab, beam_width=50, num_processes=cpu_count())
            probs = torch.softmax(logits, 2)
            beam_results, beam_scores, timesteps, out_lens = decoder.decode(probs)
            decode = [d[:l] for d,l in zip(beam_results[:, 0], out_lens[:, 0])]
            decode = [self.sentencepiece_model.DecodeIds(d.cpu().detach().numpy().tolist()) for d in decode]
        return decode

    def _compute_ctc_loss(self, logits, x, y, xlens, ylens):
        # print('torch.isfinite(logits) =',torch.isfinite(logits).all())
        logprobs = torch.log_softmax(logits.transpose(0,1), 2)
        # print('torch.isfinite(logprobs) =',torch.isfinite(logprobs).all())
        dsfactor = x.shape[1] // logprobs.shape[0]
        xlens = (xlens // dsfactor).long()
        # print(logprobs.shape, x.shape, dsfactor, xlens, ylens)
        loss = nn.CTCLoss()(logprobs, y, xlens, ylens)
        return loss

    def compute_loss(self, x, y, xlens, ylens, return_logits=True):
        logits = self.forward(x)
        loss = self._compute_ctc_loss(logits, x, y, xlens, ylens)
        if return_logits:
            return logits, loss
        return loss
