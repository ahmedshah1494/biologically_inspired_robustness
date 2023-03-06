from argparse import ArgumentParser
from importlib import import_module
import os
from time import time
from mllib.tasks.base_tasks import AbstractTask
import numpy as np
import torch
from adversarialML.biologically_inspired_models.src.runners import load_params_into_model
from adversarialML.biologically_inspired_models.src.utils import variable_length_sequence_collate_fn
import editdistance
from tqdm import tqdm

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

parser = ArgumentParser()
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--ckp', type=str)
parser.add_argument('--eps', type=float, default=0.)
args = parser.parse_args()

task = get_task_class_from_str(args.task)()
model_params = task.get_model_params()
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
for batch in tqdm(loader):
    x,y,xlens,ylens = batch
    y = y.cpu().detach().numpy().tolist()
    ylens = ylens.cpu().detach().numpy().tolist()
    with torch.no_grad():
        decode = model.decode(x)
    hyps = hyps + decode
    trans = trans + model.sentencepiece_model.DecodeIds([y_[:l] for y_,l in zip(y, ylens)])

exp_dir = os.path.dirname(os.path.dirname(args.ckp))
outfile = f'{exp_dir}/decoding_results.txt'

wers = []
with open(outfile, 'w') as f:
    for h,t in zip(hyps, trans):
        ed = editdistance.eval(h.split(), t.split())
        wer = ed/len(t.split())
        wers.append(wer)
        f.write(f'hyp: {h}\n')
        f.write(f'trn: {t}\n')
        f.write(f'wer: {wer}\n\n')

avg_wer = np.mean(wers)
print(avg_wer)