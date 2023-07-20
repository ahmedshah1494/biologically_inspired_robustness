from importlib import import_module
from typing import Tuple, Literal
import torch
import argparse
from models import _DoGFilterbank
import numpy as np
import os
from tqdm import tqdm
import webdataset as wds
from adversarialML.biologically_inspired_models.src.task_utils import LOGDIR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def create_datasets(task) -> Tuple[torch.utils.data.Dataset]:
    p = task.get_dataset_params()
    train_dataset, val_dataset, test_dataset, nclasses = p.cls.get_image_dataset(p)
    return train_dataset, val_dataset, test_dataset

def create_dataloader(task, batch_size, split: Literal['train', 'val']):
    train_dataset, val_dataset, test_dataset = create_datasets(task)
    if isinstance(train_dataset, wds.WebDataset):
        num_workers = 8 // torch.cuda.device_count()
        if split == 'train':
            dataset = train_dataset.shuffle(10_000).batched(batch_size, partial=False)
            loader = wds.WebLoader(dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=num_workers)#.with_length(len(train_dataset) // p.batch_size)
        if split == 'val':
            dataset = val_dataset.batched(batch_size, partial=False)
            loader = wds.WebLoader(dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=num_workers)#.with_length(len(val_dataset) // p.batch_size)
    else:
        if split == 'train':
            loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True, persistent_workers=True, drop_last=True)
        if split == 'val':
            loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True, persistent_workers=True, drop_last=True)
    return loader

def prune(dogfb: _DoGFilterbank, num_filters_to_keep: int, dataloader: torch.utils.data.DataLoader, early_stop_check_period=100, early_stop_patience=5, tol=0.01):
    # assumes that the loader returns either only images,
    # or that the first element in the tuple is the image
    global_freq = torch.zeros(dogfb.kernels.shape[0], device=DEVICE)
    steps_since_change = 0
    for i, batch in tqdm(enumerate(dataloader)):
        if isinstance(batch, torch.Tensor):
            x = batch
        else:
            x = batch[0]
        x = x.to(DEVICE)
        with torch.no_grad():
            e = dogfb(x)
        batch_freq = e.transpose(0,1).flatten(1,-1).sum(-1) / (dogfb.kernel_size**2)
        global_freq += batch_freq

        if (i % early_stop_check_period == 0):
            _, _selected_idxs = torch.topk(global_freq, num_filters_to_keep)
            _selected_idxs = set(_selected_idxs.detach().cpu().numpy().tolist())
            if i > 0:
                intersection = set(selected_idxs).intersection(_selected_idxs)
                pdiff = 1 - (len(intersection) / len(selected_idxs))
                if pdiff < tol:
                    print(f'Top-{num_filters_to_keep} changed by {pdiff} < {tol}. Steps since last change {steps_since_change}')
                    if steps_since_change > early_stop_patience:
                        print(f'Steps since last change {steps_since_change} > {early_stop_patience}. Early stopping at batch {i}.')
                        break
                    steps_since_change += 1
                else:
                    steps_since_change = 0
                    print(f'Top-{num_filters_to_keep} changed by {pdiff} >= {tol}. Continuing')
            selected_idxs = _selected_idxs
    freqs, selected_idxs = torch.topk(global_freq, num_filters_to_keep)
    k = dogfb.kernels.shape[0] // dogfb.stds.shape[0]
    stds = dogfb.stds[selected_idxs // k]
    corrs = dogfb.corrs[selected_idxs // k]
    kernels = dogfb.kernels[selected_idxs]
    return kernels, stds, corrs, freqs

def get_task_class_from_str(s):
    split = s.split('.')
    modstr = '.'.join(split[:-1])
    cls_name =  split[-1]
    mod = import_module(modstr)
    task_cls = getattr(mod, cls_name)
    return task_cls

def main(args):
    task = get_task_class_from_str(args.task)()
    dogfb_params = task.get_model_params()
    dogfb = dogfb_params.cls(dogfb_params).to(DEVICE)
    print(f'DoG Filterbank initialized in {dogfb.filterbank.application_mode} with {dogfb.filterbank.kernels.shape[0]} kernels')
    dataloader = create_dataloader(task, args.batch_size, args.split)
    
    odir = os.path.join(args.outdir, args.task.split('.')[-1], args.split)
    print(f'Running filter selection and saving outputs to {odir}')
    kernels, stds, corrs, freqs = prune(dogfb.filterbank, args.num_filters_to_keep, dataloader)

    if not os.path.exists(odir):
        os.makedirs(odir)
    fn = os.path.join(odir, f'{args.num_filters_to_keep}_filterbank_kernels.pt')
    torch.save(kernels.detach().cpu(), fn)
    fn = os.path.join(odir, f'{args.num_filters_to_keep}_filterbank_params.pt')
    torch.save({'stds': stds.detach().cpu(), 'corrs': corrs.detach().cpu()}, fn)
    fn = os.path.join(odir, f'{args.num_filters_to_keep}_filterbank_params.csv')
    np.savetxt(
        fn, np.concatenate([stds.detach().cpu().numpy(), corrs.detach().cpu().reshape(-1,1).numpy(), freqs.detach().cpu().reshape(-1,1).numpy()], 1),
        header= 'std_x, std_y, corr, freq'
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--split')
    parser.add_argument('--num_filters_to_keep', type=int)
    parser.add_argument('--outdir', default=LOGDIR+'/DoG_filterbanks/')
    args = parser.parse_args()

    main(args)