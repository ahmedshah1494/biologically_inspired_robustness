from argparse import ArgumentParser
from importlib import import_module
import torch
import torchvision
import numpy as np
from adversarialML.biologically_inspired_models.src.runners import load_params_into_model
from adversarialML.biologically_inspired_models.src.utils import load_pickle, load_json, write_json
from adversarialML.biologically_inspired_models.src.retina_preproc import AbstractRetinaFilter
from mllib.datasets.dataset_factory import SupportedDatasets, ImageDatasetFactory
from matplotlib import pyplot as plt
from mllib.param import BaseParameters
import matplotlib.patches as patches
import os
from itertools import product
from tqdm import tqdm
from torchattacks import APGD

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

args = parser.parse_args()

imsize = [3,224,224]

task = get_task_class_from_str(args.task)()
model_params = task.get_model_params()
model = model_params.cls(model_params)

ckp = torch.load(args.ckp)
model = load_params_into_model(ckp, model)
model = model.eval().cuda()
for m in model.modules():
    if isinstance(m, AbstractRetinaFilter):
        rblur = m
        rblur.view_scale = None
        rblur.loc_mode = 'random_in_image'
        break

dsparams = task.get_dataset_params()
_, _, test_dataset, _ = ImageDatasetFactory.get_image_dataset(dsparams)

nimgs = 5
nrows = nimgs
ncols = 4
for i,e in enumerate(test_dataset):
    if i >= nimgs:
        break
    x = e[0].cuda().unsqueeze(0)
    y = e[-1]
    y /= y.max()        

    xp = model.preprocess(x)
    yp = model(x)
    if isinstance(yp, list):
        ncols += len(yp)-1
    else:
        yp = [yp]
    if y.dim() == 1:
        y = y.reshape(int(np.sqrt(len(y))), int(np.sqrt(len(y))))
        yp = [_yp.reshape(*(y.shape)).unsqueeze(0).unsqueeze(0) for _yp in yp]
        y = y.unsqueeze(-1)
    plt.subplot(nrows, ncols, i*ncols+1)
    plt.imshow(convert_image_tensor_to_ndarray(x[0]))
    plt.subplot(nrows, ncols, i*ncols+2)
    plt.imshow(convert_image_tensor_to_ndarray(xp[0]))
    plt.subplot(nrows, ncols, i*ncols+3)
    plt.imshow(y)
    for k, _yp in enumerate(yp):
        plt.subplot(nrows, ncols, i*ncols+4+k)
        plt.imshow(convert_image_tensor_to_ndarray(_yp[0]))
plt.savefig('fixation_maps.png')