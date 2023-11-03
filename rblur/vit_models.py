from attrs import define
import torch
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from transformers import ViTForImageClassification, ViTConfig, ViTFeatureExtractor

class ViTClassifier(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        num_labels: int = 0
        intermediate_size: int = 3072
        hidden_act: str = 'gelu'
        hidden_dropout_prob: float = 0.0
        attention_probs_dropout_prob: int = 0.0
        nitializer_range: float = 0.02
        layer_norm_eps: float = 1e-12
        is_encoder_decoder: bool = False
        image_size: int = 224
        patch_size: int = 16
        num_channels: int = 3
        qkv_bias: bool = True
        encoder_stride: int = 16

    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self._make_network()

    def _make_network(self):
        cfg = ViTConfig(**(self.params.asdict()))
        self.vit = ViTForImageClassification(cfg)

    def forward(self, x):
        x = (x - 0.5)/0.5
        return self.vit(pixel_values=x)['logits']
    
    def compute_loss(self, x, y, return_logits=True):
        logits = self.forward(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        if return_logits:
            return logits, loss
        else:
            return loss
    
