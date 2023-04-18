import torch
import numpy as np
from adversarialML.biologically_inspired_models.src.trainers import LightningAdversarialTrainer, MultiAttackEvaluationTrainer
from attrs import define
from time import time

class FixationPointLightningAdversarialTrainer(LightningAdversarialTrainer):
    def forward_step(self, batch, batch_idx):
        x,y = batch
        logits, loss = self._get_outputs_and_loss(x, y)
        preds = torch.sigmoid(logits.cpu().detach()) > 0.5
        
        lr = self.scheduler.optimizer.param_groups[0]['lr']
        loss = loss.mean()
        t = time() - self.t0
        logs = {'time': t, 'lr': lr, 'accuracy': 0, 'loss': loss.detach()}
        return {'loss':loss, 'logs':logs}
    
    def test_step(self, batch, batch_idx):
        return super(LightningAdversarialTrainer, self).test_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs):
        train_metrics = {'train_accuracy': 0.}
        self.save_logs_after_test(train_metrics, outputs)
        return super(LightningAdversarialTrainer, self).test_epoch_end(outputs)

class ClickmeImportanceMapLightningAdversarialTrainer(FixationPointLightningAdversarialTrainer):
    def forward_step(self, batch, batch_idx):
        x,y = batch
        y = y.squeeze().unsqueeze(1)
        return super().forward_step((x,y), batch_idx)

    def training_step(self, batch, batch_idx):
        x,_,y = batch
        return super().training_step((x,y), batch_idx)
    
    def validation_step(self, batch, batch_idx):
        x,_,y = batch
        return super().validation_step((x,y), batch_idx)
    
    def test_step(self, batch, batch_idx):
        x,_,y = batch
        return super().test_step((x,y), batch_idx)

class RetinaFilterWithFixationPredictionLightningAdversarialTrainer(LightningAdversarialTrainer):
    @define(slots=False)    
    class TrainerParams(LightningAdversarialTrainer.TrainerParams):
        loc_sampling_temp_decay_rate: float = 1.
    
    def __init__(self, params: TrainerParams, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.tau = 0.

    def decay_temperature(self):
        if 'RetinaFilterWithFixationPrediction' not in locals():
            from adversarialML.biologically_inspired_models.src.fixation_prediction.models import RetinaFilterWithFixationPrediction
        for m in self.model.modules():
            if isinstance(m, RetinaFilterWithFixationPrediction):
                m.params.loc_sampling_temp *= self.params.loc_sampling_temp_decay_rate
                break
        self.tau = m.params.loc_sampling_temp
        return self.tau
    
    def forward_step(self, batch, batch_idx):
        outputs = super().forward_step(batch, batch_idx)
        outputs['logs']['tau'] = self.tau
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = super().training_step(batch, batch_idx)
        self.decay_temperature()
        return outputs