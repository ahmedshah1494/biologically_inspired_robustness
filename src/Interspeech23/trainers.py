from adversarialML.biologically_inspired_models.src.trainers import LightningAdversarialTrainer
from time import time

class SpeechAdversarialTrainer(LightningAdversarialTrainer):
    def _get_outputs_and_loss(self, x, y, xlens, ylens):
        return self.model.compute_loss(x, y, xlens, ylens, return_logits=True)

    def forward_step(self, batch, batch_idx):
        x,y, xlens, ylens = batch
        logits, loss = self._get_outputs_and_loss(*batch)
        lr = self.scheduler.optimizer.param_groups[0]['lr']
        loss = loss.mean()
        t = time() - self.t0
        logs = {'time': t, 'lr': lr, 'accuracy':0., 'loss': loss.detach()}
        return {'loss':loss, 'logs':logs}
    
    def _maybe_attack_batch(self, batch, adv_attack):
        return batch

    