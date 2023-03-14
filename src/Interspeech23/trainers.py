from enum import Enum, auto
from attrs import define, field
import numpy as np
from adversarialML.biologically_inspired_models.src.trainers import LightningAdversarialTrainer, ActivityOptimizationParams, ActivityOptimizationSchedule
from adversarialML.biologically_inspired_models.src.models import ConsistencyOptimizationMixin
from time import time
from torchmetrics.functional import word_error_rate

class SpeechAdversarialTrainer(LightningAdversarialTrainer):
    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.total_utt_time = 0

    def _get_outputs_and_loss(self, x, y, xlens, ylens):
        return self.model.compute_loss(x, y, xlens, ylens, return_logits=True)

    def forward_step(self, batch, batch_idx):
        x,y, xlens, ylens = batch
        logits, loss = self._get_outputs_and_loss(*batch)
        lr = self.scheduler.optimizer.param_groups[0]['lr']
        loss = loss.mean()
        t = time() - self.t0
        # self.total_utt_time += (sum(xlens) / 16_000)/3600
        if not self.training:
            decode, _ = self.model.decode(x, return_idxs=True)
            y = y.cpu().detach().numpy().tolist()
            ylens = ylens.cpu().detach().numpy().tolist()
            y = [y_[:l] for y_,l in zip(y, ylens)]
            trans = self.model.sentencepiece_model.DecodeIds(y)
            wer = word_error_rate(decode, trans)
        else:
            wer = 0.
        logs = {'time': t, 'lr': lr, 'accuracy':wer, 'loss': loss.detach()}
        return {'loss':loss, 'logs':logs}
    
    def _maybe_attack_batch(self, batch, adv_attack):
        return batch

class ConsistentActivationSpeechAdversarialTrainer(SpeechAdversarialTrainer):
    @define(slots=False)
    class TrainerParams(SpeechAdversarialTrainer.TrainerParams):
        act_opt_params: ActivityOptimizationParams = field(factory=ActivityOptimizationParams)

    def __init__(self, params: TrainerParams, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.params = params
        self._load_max_act_opt_lrs()

    def _load_max_act_opt_lrs(self):
        self.max_act_opt_lrs = {}
        for n, m in self.model.named_modules():
            if isinstance(m, ConsistencyOptimizationMixin):
                self.max_act_opt_lrs[n] = m.act_opt_step_size

    def _update_act_opt_lrs(self, epoch_idx: int):
        if (epoch_idx < self.params.act_opt_params.num_warmup_epochs) and self.params.act_opt_params.act_opt_lr_warmup_schedule != ActivityOptimizationSchedule.CONST:
            for n,m in self.model.named_modules():
                if n in self.max_act_opt_lrs:
                    init_lr = min(self.params.act_opt_params.init_act_opt_lr, self.max_act_opt_lrs[n])
                    if self.params.act_opt_params.act_opt_lr_warmup_schedule == ActivityOptimizationSchedule.GEOM:
                        m.act_opt_step_size = np.geomspace(init_lr, 
                                                            self.max_act_opt_lrs[n], 
                                                            self.params.act_opt_params.num_warmup_epochs)[epoch_idx]
                    if self.params.act_opt_params.act_opt_lr_warmup_schedule == ActivityOptimizationSchedule.LINEAR:
                        m.act_opt_step_size = np.linspace(init_lr, 
                                                            self.max_act_opt_lrs[n], 
                                                            self.params.act_opt_params.num_warmup_epochs)[epoch_idx]
                    print(n, m.act_opt_step_size)
    def on_train_epoch_start(self) -> None:
        self._update_act_opt_lrs(self.current_epoch)
        return super().on_train_epoch_start()