from argparse import ArgumentParser
from importlib import import_module
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
        # rblur.params.loc_mode = 'center'
        # break
    if isinstance(m, FixationPredictionNetwork):
        model = m

dsparams = task.get_dataset_params()
test_dataset = ImageDatasetFactory.get_image_dataset(dsparams)[2]

nimgs = 5
nrows = nimgs
ncols = 5
if hasattr(test_dataset, 'dataset') and isinstance(test_dataset.dataset, (torchvision.datasets.ImageFolder, ImagenetFileListDataset)):
    rand_idxs = np.random.permutation(len(test_dataset))
    test_dataset = torch.utils.data.Subset(test_dataset, rand_idxs)
    # test_dataset.samples = test_dataset.samples[rand_idxs]
    # test_dataset.targets = test_dataset.targets[rand_idxs]
    # test_dataset.imgs = test_dataset.imgs[rand_idxs]
count = 0
for e in test_dataset:
    x = e[0].cuda().unsqueeze(0)
    y = e[-1]
    if (y == 1).all() or (y == 0).all():
        continue
    y /= y.max()
    xp = model.preprocess(x)
    yp = model(x)
    if hasattr(model, 'standardize_heatmap'):
        yp = torch.relu(model.standardize_heatmap(yp))
    if getattr(model_params, 'loss_fn', None) == 'bce':
        yp = torch.sigmoid(yp)
    if isinstance(yp, list):
        ncols += len(yp)-1
    else:
        yp = [yp]
    if y.dim() == 1:
        y = y.reshape(int(np.sqrt(len(y))), int(np.sqrt(len(y))))
        y = y.unsqueeze(-1)
    if yp[0].dim() == 1:
        yp = [_yp.reshape(*(y.shape[:-1])).unsqueeze(0).unsqueeze(0) for _yp in yp]
    plt.subplot(nrows, ncols, count*ncols+1)
    plt.imshow(convert_image_tensor_to_ndarray(x[0]))
    plt.subplot(nrows, ncols, count*ncols+2)
    plt.imshow(convert_image_tensor_to_ndarray(xp[0]))
    plt.subplot(nrows, ncols, count*ncols+3)
    plt.imshow(y)
    print(y.shape, y.min(), y.max())
    for k, _yp in enumerate(yp):
        plt.subplot(nrows, ncols, count*ncols+4+k)
        plt.imshow(convert_image_tensor_to_ndarray(_yp[0]))
        print(_yp[0].shape, _yp[0].min(), _yp[0].max())
        plt.subplot(nrows, ncols, count*ncols+5+k)
        _yp = torch.nn.functional.interpolate(_yp, size=y.shape[:2])
        plt.imshow(convert_image_tensor_to_ndarray(_yp[0]))
    count += 1
    if count >= nimgs:
        break

exp_num = args.ckp.split('/')[-3]
plt.savefig(f'fixation_prediction/{args.task.split(".")[-1]}-{exp_num}_fixation_maps.png')