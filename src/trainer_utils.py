from mllib.utils.trainer_utils import _move_tensors_to_device
import torch

def _maybe_attack_batch(batch, adv_attack):
    x,y = batch
    if adv_attack is not None:
        x = adv_attack(x, y)
    batch = (x,y)
    return batch

def _make_adv_datset(loader, adv_attack, device='cpu'):
    X_adv = []
    Y = []
    for x, y in loader:
        x,y = _move_tensors_to_device((x,y), device)
        if adv_attack is None:
            xadv = x
        else:
            xadv = adv_attack.perturb(x, y)
        X_adv.append(xadv.detach().cpu())
        Y.append(y.detach().cpu())
    X_adv = torch.cat(X_adv, dim=0)
    Y = torch.cat(Y, dim=0)
    dataset =  torch.utils.data.TensorDataset(X_adv, Y)
    return dataset, (X_adv, Y)

def make_dataloader(dataset, batch_size, shuffle=False, device='cpu'):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=(device == 'cuda'))
    return loader