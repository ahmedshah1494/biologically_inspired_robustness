from argparse import ArgumentParser
from collections import Counter
from enum import Enum, auto
from importlib import import_module
import torch
import torchvision
import numpy as np
from rblur.runners import load_params_into_model
from rblur.utils import load_pickle, load_json, write_json
from rblur.retina_preproc import AbstractRetinaFilter
from rblur.fixation_prediction.models import DeepGazeIIE
from mllib.datasets.dataset_factory import SupportedDatasets
from mllib.datasets.imagenet_filelist_dataset import ImagenetFileListDataset
from deepgaze_pytorch import deepgaze_pytorch
from matplotlib import pyplot as plt
from mllib.param import BaseParameters
import matplotlib.patches as patches
import os
from itertools import product
from tqdm import tqdm
from torchattacks import APGD
from PIL import Image

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

class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()
    
class SupportedModels(AutoName):
    DEEPGAZE2E = auto()
    # DEEPGAZE3 = auto()
SupportedModels.DEEPGAZE2E
parser = ArgumentParser()
parser.add_argument('--model', required=True, type=lambda k: SupportedModels._value2member_map_[k], choices=SupportedModels)
parser.add_argument('--input_image_size', type=int, default=1024)
parser.add_argument('--output_image_size', type=int, default=320)
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--num_test', type=int, default=np.inf)
parser.add_argument('--image_dir', type=str)
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--output_dir', type=str)
parser.add_argument('--overwrite', action='store_true')

args = parser.parse_args()
print(args.model, SupportedModels.DEEPGAZE2E, args.model == SupportedModels.DEEPGAZE2E)
if args.model == SupportedModels.DEEPGAZE2E:
    model = DeepGazeIIE(True)
model = model.cuda()
# elif args.model == SupportedModels.DEEPGAZE3:
#     model = deepgaze_pytorch.DeepGazeIII(True)


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(args.input_image_size-1, max_size=args.input_image_size),
    # torchvision.transforms.CenterCrop(args.image_size),
    torchvision.transforms.ToTensor()
])
class mDataset(ImagenetFileListDataset):
    # def __init__(self, root: str, split='train', transforms = None, transform = None, target_transform = None) -> None:
    #     super().__init__(root, split, transforms, transform, target_transform)
    #     filesizes = [Image.open(fn).size for fn in self.samples]
    #     filesizes = [max(fsz)/min(fsz) for fsz in filesizes]
    #     fszcounter = Counter(filesizes)
    #     print(fszcounter, max(fszcounter.keys()))
    #     sorted_idxs = sorted(range(len(self.filesizes)), key=lambda i: fszcounter[filesizes[i]], reverse=True)
    #     self.samples = self.samples[sorted_idxs]
    #     self.targets = self.targets[sorted_idxs]

    def __getitem__(self, i):
        fn = self.samples[i]
        x, y = super().__getitem__(i)
        return fn, x, y
test_dataset = mDataset(args.image_dir, split=args.split, transform=transform)
nclasses = len(set(test_dataset.targets))
test_dataset = torch.utils.data.Subset(test_dataset, range(args.start_idx, args.start_idx+args.num_test))
print(len(test_dataset))
loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

correct = 0
adv_correct = 0
total = 0
t = tqdm(loader)

for i,batch in enumerate(t):
    # if not ((i > args.start_idx) and (i < args.start_idx + args.num_test)):
    #     print(i, args.start_idx, args.num_test, 'skipping')
    #     continue
    filenames, x, y = batch
    print(x.shape)
    filenames = np.array(filenames)
    idx_to_include = []
    if not args.overwrite:
        for i, fn in enumerate(filenames):
            [label, fn] = fn.split('/')[-2:]
            odir = f'{args.output_dir}/{args.model.value}/{args.split}/{label}/'
            ofn = f'{odir}/{fn.split(".")[0]}.npz'
            if not os.path.exists(ofn):
                idx_to_include.append(i)
        filenames, x, y = filenames[idx_to_include], x[idx_to_include], y[idx_to_include]
    with torch.no_grad():
        fixation_maps = model(x.cuda()).cpu().detach()
        fixation_maps = torchvision.transforms.functional.resize(fixation_maps, args.output_image_size-1, max_size=args.output_image_size).numpy().astype('float16')

    print('saving logit maps...')
    for fn, fmap, l in zip(filenames, fixation_maps, y):
        l = int(l)
        [label, fn] = fn.split('/')[-2:]
        odir = f'{args.output_dir}/{args.model.value}/{args.split}/{label}/'
        if not os.path.exists(odir):
            os.makedirs(odir)
        fn = fn.split('.')[0]
        np.savez(f'{odir}/{fn}.npz', fixation_probs=fmap, label=l)