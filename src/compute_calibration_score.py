from argparse import ArgumentParser
from importlib import import_module
import os
import torch
import torchvision
import numpy as np
from adversarialML.biologically_inspired_models.src.runners import load_params_into_model
from adversarialML.biologically_inspired_models.src.utils import load_pickle, load_json, write_json
from adversarialML.biologically_inspired_models.src.retina_preproc import AbstractRetinaFilter
from adversarialML.biologically_inspired_models.src.fixation_prediction.models import FixationPredictionNetwork
from mllib.datasets.dataset_factory import SupportedDatasets, ImageDatasetFactory
from mllib.datasets.imagenet_filelist_dataset import ImagenetFileListDataset
from matplotlib import pyplot as plt
from mllib.param import BaseParameters
from tqdm import tqdm

def get_task_class_from_str(s):
    split = s.split('.')
    modstr = '.'.join(split[:-1])
    cls_name =  split[-1]
    mod = import_module(modstr)
    task_cls = getattr(mod, cls_name)
    return task_cls

def convert_image_tensor_to_ndarray(img):
    return img.cpu().detach().transpose(0,1).transpose(1,2).numpy()

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

parser = ArgumentParser()
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--ckp', type=str, required=True)
parser.add_argument('--use_common_corruption_testset', action='store_true')

args = parser.parse_args()

imsize = [3,224,224]

task = get_task_class_from_str(args.task)()
model_params = task.get_model_params()
model = model_params.cls(model_params)

ckp = torch.load(args.ckp)
model = load_params_into_model(ckp, model)
model = model.eval().cuda()
print(model)
for m in model.modules():
    if isinstance(m, AbstractRetinaFilter):
        rblur = m
        rblur.view_scale = None
        rblur.params.loc_mode = 'center'

dsparams = task.get_dataset_params()
dsparams.max_num_test = 100_0000
if args.use_common_corruption_testset:
    if dsparams.dataset == SupportedDatasets.ECOSET10:
        dsparams.dataset = SupportedDatasets.ECOSET10C_FOLDER
        dsparams.datafolder = os.path.join(os.path.dirname(os.path.dirname(dsparams.datafolder)), 'distorted')
    if dsparams.dataset == SupportedDatasets.ECOSET100_FOLDER:
        dsparams.dataset = SupportedDatasets.ECOSET100C_FOLDER
        dsparams.datafolder = os.path.join(dsparams.datafolder, 'distorted')
    if dsparams.dataset == SupportedDatasets.ECOSET_FOLDER:
        dsparams.dataset = SupportedDatasets.ECOSETC_FOLDER
        dsparams.datafolder = '/home/mshah1/workhorse3/ecoset/distorted'
    if dsparams.dataset == SupportedDatasets.CIFAR10:
        dsparams.dataset = SupportedDatasets.CIFAR10C
        dsparams.datafolder = '/home/mshah1/workhorse3/cifar-10-batches-py/distorted'
_, _, test_dataset, _ = ImageDatasetFactory.get_image_dataset(dsparams)
loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

labels = []
cls_probs = []
for x,y in tqdm(loader):
    with torch.no_grad():
        logits = model(x.cuda()).detach().cpu()
    probs = torch.softmax(logits, 1)
    labels.append(y)
    cls_probs.append(probs)
labels = torch.cat(labels, 0).detach().cpu().numpy()
cls_probs = torch.cat(cls_probs, 0).detach().cpu().numpy()

preds = np.argmax(cls_probs, 1)
acc = (preds == labels).astype(float).mean()
print(f'Accuracy = {acc}')

y = np.repeat(labels, cls_probs.shape[1])
q = cls_probs.flatten()
yp = np.arange(len(q)) % cls_probs.shape[1]
sorted_idxs = sorted(range(len(q)), key=lambda i: q[i])
q = q[sorted_idxs]
y = y[sorted_idxs]
yp = yp[sorted_idxs]

assert len(y) == len(q)

binsz = 1000 # pick a bin size that exactly divides the data
q = q.reshape(-1, binsz)
y = y.reshape(-1, binsz)
yp = yp.reshape(-1, binsz)

py = (y == yp).astype(float).mean(1)
pq = q.mean(1)
print(np.stack([py, pq, (py - pq)**2], 1))
rmsce = np.sqrt(((py - pq) ** 2).mean())
print(f'RMS Callibration Error = {rmsce}')