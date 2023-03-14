from argparse import ArgumentParser
from importlib import import_module
import os
from time import time
from mllib.tasks.base_tasks import AbstractTask
from mllib.param import BaseParameters
import numpy as np
import torch
from adversarialML.biologically_inspired_models.src.runners import load_params_into_model
from adversarialML.biologically_inspired_models.src.utils import variable_length_sequence_collate_fn
import editdistance
from tqdm import tqdm
from torchmetrics.functional import word_error_rate

def load_model_from_ckpdir(d):
    files = os.listdir(d)
    model_file = [f for f in files if f.startswith('model')][0]
    model = torch.load(os.path.join(d, model_file))
    return model

def get_task_class_from_str(s):
    split = s.split('.')
    modstr = '.'.join(split[:-1])
    cls_name =  split[-1]
    mod = import_module(modstr)
    task_cls = getattr(mod, cls_name)
    return task_cls

def set_param(p: BaseParameters, param, value):
    if isinstance(p, BaseParameters):
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

parser = ArgumentParser()
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--ckp', type=str)
parser.add_argument('--num_test', type=int, default=3000)
parser.add_argument('--eps', type=float, default=0.)
parser.add_argument('--alpha', type=float, default=0.)
parser.add_argument('--beta', type=float, default=0.)
parser.add_argument('--beam_width', type=int, default=100)
parser.add_argument('--kenLM_path', type=str)
args = parser.parse_args()

task = get_task_class_from_str(args.task)()
model_params = task.get_model_params()

set_param(model_params, 'decoding_alpha', args.alpha)
set_param(model_params, 'decoding_beta', args.beta)
set_param(model_params, 'decoding_beam_width', args.beam_width)
if args.kenLM_path:
    set_param(model_params, 'kenlm_model_path', args.kenLM_path)

model = model_params.cls(model_params)

print('loading checkpoint...')
ckp = torch.load(args.ckp)
model = load_params_into_model(ckp, model)
model = model.eval()

dsparams = task.get_dataset_params()
test_dataset = dsparams.cls.get_image_dataset(dsparams)[2]
loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=8, collate_fn=variable_length_sequence_collate_fn, shuffle=False)

hyps = []
trans = []
Y = []
all_decoded_idxs = []
for batch in tqdm(loader):
    x,y,xlens,ylens = batch
    y = y.cpu().detach().numpy().tolist()
    ylens = ylens.cpu().detach().numpy().tolist()
    y = [y_[:l] for y_,l in zip(y, ylens)]
    with torch.no_grad():
        decode, decoded_idxs = model.decode(x, return_idxs=True)
    all_decoded_idxs = all_decoded_idxs + decoded_idxs
    Y = Y + y
    hyps = hyps + decode
    trans = trans + model.sentencepiece_model.DecodeIds(y)
    if len(hyps) >= args.num_test:
        break
print(len(hyps), len(trans), len(Y), len(all_decoded_idxs))
exp_dir = os.path.dirname(os.path.dirname(args.ckp))
outfile = f'{exp_dir}/decoding_results_n={args.num_test}_a={args.alpha}_b={args.beta}_w={args.beam_width}_lm={os.path.basename(args.kenLM_path).split(".")[0] if args.kenLM_path else "na"}.txt'
print(f'writing to {outfile}')
wers = []
cers = []
with open(outfile, 'w') as f:
    for h,t,didx,y in zip(hyps, trans, all_decoded_idxs, Y):
        # wed = editdistance.eval(h.split(), t.split())
        # wer = wed/len(t.split())
        wer = word_error_rate([h],[t])
        ced = editdistance.eval(h, t)
        cer = ced/len(t)
        ied = editdistance.eval(didx, y)
        ier = ied/len(y)
        wers.append(wer)
        cers.append(cer)
        f.write(f'hyp: {h}\n')
        f.write(f'trn: {t}\n')
        f.write(f'ier: {ier}\n')
        f.write(f'cer: {cer}\n')
        f.write(f'wer: {wer}\n\n')
print(f'avg_wer:{np.mean(wers)}\tavg_cer:{np.mean(cers)}')