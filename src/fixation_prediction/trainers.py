import torch
import numpy as np
from adversarialML.biologically_inspired_models.src.trainers import LightningAdversarialTrainer, MultiAttackEvaluationTrainer
from adversarialML.biologically_inspired_models.src.fixation_prediction.models import RetinaFilterWithFixationPrediction
from mllib.trainers.base_trainers import PytorchLightningTrainer
# from pysaliency.roc import general_roc
# from pysaliency.numba_utils import auc_for_one_positive
from mllib.utils.metric_utils import compute_accuracy
from mllib.param import BaseParameters
from attrs import define
from time import time

def set_param(p:BaseParameters, param, value):
    if hasattr(p, param):
        setattr(p, param, value)
    else:
        d = p.asdict(recurse=False)
        for v in d.values():
            if isinstance(v, BaseParameters):
                set_param(v, param, value)
            elif np.iterable(v):
                for x in v:
                    if isinstance(x, BaseParameters):
                        set_param(x, param, value)
    return p
class FixationPointLightningAdversarialTrainer(LightningAdversarialTrainer):
    def forward_step(self, batch, batch_idx):
        x,y = batch
        logits, loss = self._get_outputs_and_loss(x, y)
        
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
    
class PrecomputedFixationMapMultiAttackEvaluationTrainer(MultiAttackEvaluationTrainer):
    @define(slots=False)
    class TrainerParams(MultiAttackEvaluationTrainer.TrainerParams):
        set_fixation_to_max: bool = True

    def get_retina_fixation_module(self) -> RetinaFilterWithFixationPrediction:
        for m in self.model.modules():
            if isinstance(m, RetinaFilterWithFixationPrediction):                
                return m
        return None
    
    def test_step(self, batch, batch_idx):
        x,y,m = batch
        m = m.squeeze(-1)
        while m.dim() < 4:
            m = m.unsqueeze(1)
        b, w = m.shape[0], m.shape[-1]
        rfmodule = self.get_retina_fixation_module()
        if rfmodule and rfmodule.params.salience_map_provided_as_input_channel:
            x = torch.cat([x,m], dim=1)
        elif self.params.set_fixation_to_max:
            m = m.reshape(b, -1)
            locs = m.argmax(1).cpu().detach().numpy()
            rowi = locs // w
            coli = locs % w
            locs = [(ri, ci) for ri,ci in zip(rowi, coli)]
            set_param(self.model.params, 'loc_mode', 'const')
            set_param(self.model.params, 'loc', locs)
            set_param(self.model.params, 'batch_size', 1)
        # print(self.model.feature_model.layers[1].params.loc_mode)
        return super().test_step((x,y), batch_idx)

class RetinaFilterWithFixationPredictionLightningAdversarialTrainer(LightningAdversarialTrainer):
    @define(slots=False)    
    class TrainerParams(LightningAdversarialTrainer.TrainerParams):
        loc_sampling_temp_decay_rate: float = 1.
    
    def __init__(self, params: TrainerParams, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.tau = 0.
    
    def get_retina_fixation_module(self) -> RetinaFilterWithFixationPrediction:
        for m in self.model.modules():
            if isinstance(m, RetinaFilterWithFixationPrediction):                
                break
        return m
    def decay_temperature(self):
        m = self.get_retina_fixation_module()
        m.params.loc_sampling_temp *= self.params.loc_sampling_temp_decay_rate
        self.tau = m.params.loc_sampling_temp
        return self.tau
    
    def forward_step(self, batch, batch_idx):
        x,y,m = batch
        rfmodule = self.get_retina_fixation_module()
        if rfmodule.params.salience_map_provided_as_input_channel:
            x = torch.cat([x,m], dim=1)
        logits, loss = self._get_outputs_and_loss(x, y)
        acc, _ = compute_accuracy(logits.detach(), y.detach())
        
        lr = self.scheduler.optimizer.param_groups[0]['lr']
        loss = loss.mean()
        t = time() - self.t0
        logs = {'time': t, 'lr': lr, 'accuracy': acc, 'loss': loss.detach(), 'tau': self.tau}
        outputs = {'loss':loss, 'logs':logs}
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = super().training_step(batch, batch_idx)
        self.decay_temperature()
        return outputs
    
class RetinaFilterWithFixationPredictionMultiAttackEvaluationTrainer(MultiAttackEvaluationTrainer):
    def get_retina_fixation_module(self) -> RetinaFilterWithFixationPrediction:
        for m in self.model.modules():
            if isinstance(m, RetinaFilterWithFixationPrediction):                
                break
        return m
    
    def test_step(self, batch, batch_idx):
        if len(batch) == 3:
            x,y,m = batch
        else:
            x, y = batch
        rfmodule = self.get_retina_fixation_module()
        if rfmodule.params.salience_map_provided_as_input_channel:
            x = torch.cat([x,m], dim=1)
            return super().test_step((x,y), batch_idx)
        else:
            m = rfmodule(x, return_fixation_maps=True)[1][0]
            rfmodule.params.salience_map_provided_as_input_channel = True
            K = rfmodule.params.num_train_fixation_points
            rfmodule.params.num_train_fixation_points = 1
            x = torch.cat([x,m], dim=1)
            output = super().test_step((x,y), batch_idx)
            rfmodule.params.salience_map_provided_as_input_channel = False
            rfmodule.params.num_train_fixation_points = K
            return output

        