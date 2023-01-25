from argparse import ArgumentParser
from importlib import import_module
import torch
import numpy as np
from adversarialML.biologically_inspired_models.src.runners import load_params_into_model
from adversarialML.biologically_inspired_models.src.utils import load_pickle
from adversarialML.biologically_inspired_models.src.retina_preproc import AbstractRetinaFilter
from mllib.datasets.dataset_factory import SupportedDatasets
from matplotlib import pyplot as plt
from mllib.param import BaseParameters
from matplotlib import patches
import os

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
parser.add_argument('--ckp', type=str)
args = parser.parse_args()

task = get_task_class_from_str(args.task)()
model_params = task.get_model_params()
model = model_params.cls(model_params)

ckp = torch.load(args.ckp)
model = load_params_into_model(ckp, model)
model = model.eval()
for m in model.modules():
    if isinstance(m, AbstractRetinaFilter):
        rblur = m
        break
rblur.view_scale = None
x = torch.rand(1,3,224,224, requires_grad=True)

ds_params = task.get_dataset_params()
ds_params.dataset = SupportedDatasets.ECOSET10wBB_FOLDER
ds_params.datafolder = os.path.dirname(os.path.dirname(ds_params.datafolder))
_, _, test_dataset, _ = ds_params.cls.get_image_dataset(ds_params)

nimgs = 5
idxs = np.random.permutation(len(test_dataset))
ncol = 2
i = 0
j = 0
while i < nimgs:
    x, y, bb = test_dataset[j]
    j += 1
    if (bb == 0.5).all():
        continue
    x = x.unsqueeze(0)
    bb = bb.detach().cpu().numpy()
    l = (800 - int(x.shape[2]*(bb[0] + bb[2])/2), 800 - int(x.shape[3]*(bb[1] + bb[3])/2))
    print(l)
    px = rblur._forward_batch(x, loc_idx=l)
    print(torch.norm(x[0] - px[0], p=2, dim=[0,1,2]))
    ax = plt.subplot(nimgs , ncol, i*ncol + 1)
    plt.imshow(convert_image_tensor_to_ndarray(torch.clamp(x[0],0.,1.).float()))
    xmin, ymin, xmax, ymax = bb[:4]
    rect = patches.Rectangle((xmin, xmax), xmax-xmin, ymax-ymin, linewidth=0.5, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    plt.xticks([],[])
    plt.yticks([],[])
    plt.subplot(nimgs , ncol, i*ncol + 2)    
    plt.xticks([],[])
    plt.yticks([],[])
    plt.imshow(convert_image_tensor_to_ndarray(torch.clamp(px[0],0.,1.).float()))
    i += 1
plt.tight_layout()
plt.savefig('rblur_img.png')