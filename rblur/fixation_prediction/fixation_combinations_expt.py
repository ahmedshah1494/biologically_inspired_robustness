from argparse import ArgumentParser
from importlib import import_module
import torch
import torchvision
import numpy as np
from adversarialML.biologically_inspired_models.src.runners import load_params_into_model
from adversarialML.biologically_inspired_models.src.utils import load_pickle, load_json, write_json
from adversarialML.biologically_inspired_models.src.retina_preproc import AbstractRetinaFilter
from mllib.datasets.dataset_factory import SupportedDatasets
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
parser.add_argument('--eps', type=float, default=0.)
parser.add_argument('--num_test', type=int, default=np.inf)
parser.add_argument('--N', type=int, default=49)
parser.add_argument('--image_dir', type=str)
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--logit_map_output_dir', type=str)
parser.add_argument('--overwrite', action='store_true')

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
        break
rblur.view_scale = None
x = torch.rand(1, *imsize, requires_grad=True)

vf_rad = 800
col_locs = np.linspace(0, imsize[1]-1, int(np.sqrt(args.N)), dtype=np.int32)
row_locs = np.linspace(0, imsize[2]-1, int(np.sqrt(args.N)), dtype=np.int32)
locs = list(product(col_locs, row_locs))

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor()
])
class mDataset(ImagenetFileListDataset):
    def __getitem__(self, i):
        fn = self.samples[i]
        x, y = super().__getitem__(i)
        return fn, x, y
test_dataset = mDataset(args.image_dir, split=args.split, transform=transform)
nclasses = len(set(test_dataset.targets))

loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

correct = 0
adv_correct = 0
total = 0
t = tqdm(loader)

for i,batch in enumerate(t):
    # if i < 473:
    #     continue
    filenames, x, y = batch
    filenames = np.array(filenames)
    idx_to_include = []
    if not args.overwrite:
        for i, fn in enumerate(filenames):
            [label, fn] = fn.split('/')[-2:]
            odir = f'{args.logit_map_output_dir}/{label}/'
            ofn = f'{odir}/{fn.split(".")[0]}.npz'
            if not os.path.exists(ofn):
                idx_to_include.append(i)
        filenames, x, y = filenames[idx_to_include], x[idx_to_include], y[idx_to_include]

    # print(filenames)
    # exit()
    print(x.shape)
    x = x.cuda()
    total += x.shape[0]
    # logit_map = np.zeros((x.shape[0], nclasses, x.shape[2], x.shape[3]))
    loc_logits = []
    loc_correct = []
    loc_probs = []
    c = np.zeros((x.shape[0],), dtype=bool)
    for l in locs:
        set_param(model.params, 'loc_mode', 'const')
        # set_param(model.params, 'loc', (vf_rad - l[0], vf_rad - l[1]))
        set_param(model.params, 'loc', (l[0], l[1]))
        if args.eps > 0:
            x = APGD(model, eps=args.eps)(x, y)
        logits = model(x).detach().cpu()
        yp = torch.argmax(logits,1)
        c_ = (y == yp).numpy()
        c |= c_
        loc_logits.append(logits)
        loc_correct.append(c_.astype(int))
        loc_prob = torch.softmax(logits, 1)[torch.arange(len(y)), y].numpy()
        loc_probs.append(loc_prob)
        # logit_map[..., l[0], l[1]] = logits.numpy()
        print(l, x.shape, y.shape, c.astype(int).sum())
    loc_logits = np.stack(loc_logits, 1)
    loc_correct = np.stack(loc_correct, 1)
    loc_probs = np.stack(loc_probs, 1)

    print('saving logit maps...')
    for fn, llogits, lcor, lprob, l in zip(filenames, loc_logits, loc_correct, loc_probs, y):
        l = int(l)
        [label, fn] = fn.split('/')[-2:]
        odir = f'{args.logit_map_output_dir}/{label}/'
        if not os.path.exists(odir):
            os.makedirs(odir)
        fn = fn.split('.')[0]
        # cmap = (np.argmax(llogits, 0) == l).astype(int)
        np.savez(f'{odir}/{fn}.npz', fixation_logits=llogits, fixation_correct=lcor, fixation_probs=lprob, label=l, locs=locs)
    correct += c.astype(int).sum()
    accuracy = correct/total
    # imgs.append(x.cpu().detach().numpy())
    # labels.append(y.cpu().detach().numpy())
    t.set_postfix({'accuracy':accuracy})
    # break