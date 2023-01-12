from argparse import ArgumentParser
from importlib import import_module
import torch
import numpy as np
from adversarialML.biologically_inspired_models.src.runners import load_params_into_model
from adversarialML.biologically_inspired_models.src.utils import load_pickle, load_json, write_json
from adversarialML.biologically_inspired_models.src.retina_preproc import AbstractRetinaFilter
from mllib.datasets.dataset_factory import SupportedDatasets
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
parser.add_argument('--plot_examples', action='store_true')
parser.add_argument('--num_examples', type=int, default=9)

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
# locs = [(111,111)]

ds_params = task.get_dataset_params()
# ds_params.dataset = SupportedDatasets.ECOSET10_FOLDER
# ds_params.datafolder = os.path.dirname(os.path.dirname(ds_params.datafolder))
ds_params.dataset = SupportedDatasets.ECOSET100_FOLDER
ds_params.datafolder = f'{os.path.dirname(ds_params.datafolder)}/eval_dataset_dir'
ds_params.max_num_test = args.num_test
print(ds_params)
_, _, test_dataset, nclasses = ds_params.cls.get_image_dataset(ds_params)

if not args.plot_examples:
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    correct = 0
    adv_correct = 0
    total = 0
    t = tqdm(loader)

    imgs = []
    logit_maps = []
    labels = []
    preds = []
    for x,y in t:
        x = x.cuda()
        total += x.shape[0]
        lmap = np.zeros((x.shape[0], nclasses, x.shape[2], x.shape[3]))
        c = np.zeros((x.shape[0],), dtype=bool)
        for l in locs:
            if not c.all():
                if not args.plot_examples:
                    x_ = x[~c]
                    y_ = y[~c]
                else:
                    x_ = x
                    y_ = y

                set_param(model.params, 'loc_mode', 'const')
                set_param(model.params, 'loc', (vf_rad - l[0], vf_rad - l[1]))
                if args.eps > 0:
                    x_ = APGD(model, eps=args.eps)(x_, y_)
                logits = model(x_).detach().cpu()
                yp = torch.argmax(logits,1)
                # print(y.shape, y_.shape, yp.shape, c.astype(int).sum())
                c_ = (y_ == yp).numpy()
                c[~c] |= c_
                if args.plot_examples:
                    lmap[..., l[0], l[1]] = logits.numpy()
                print(l, x_.shape, y_.shape, c.astype(int).sum())
        correct += c.astype(int).sum()
        accuracy = correct/total
        logit_maps.append(lmap)
        imgs.append(x.cpu().detach().numpy())
        labels.append(y.cpu().detach().numpy())
        t.set_postfix({'accuracy':accuracy})

    ofn = f'{os.path.dirname(os.path.dirname(args.ckp))}/many_fixations_results.json'
    print(f'writing results to {ofn}')
    if os.path.exists(ofn):
        results = load_json(ofn)
    else:
        results = {}
    results[f'accuracy_eps={args.eps}_N={args.N}'] = accuracy
    write_json(results, ofn)

if args.plot_examples:
    idxs = np.random.choice(len(test_dataset), args.num_examples, replace=False)
    x, y = list(zip(*([test_dataset[i] for i in idxs])))
    x = torch.stack(x, 0).cuda()
    y = torch.LongTensor(y)

    logit_map = np.zeros((x.shape[0], nclasses, x.shape[2], x.shape[3]))
    pred_map = np.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=np.int32)
    for l in locs:
        set_param(model.params, 'loc_mode', 'const')
        set_param(model.params, 'loc', (vf_rad - l[0], vf_rad - l[1]))
        if args.eps > 0:
            xadv = APGD(model, eps=args.eps)(x, y)
        else:
            xadv = x
        logits = model(xadv).detach().cpu()
        yp = torch.argmax(logits, 1)
        logit_map[..., l[0], l[1]] = logits
        pred_map[..., l[0], l[1]] = yp

    nrows = ncols = int(np.sqrt(args.num_examples))
    plt.figure()
    step_size = row_locs[1] - row_locs[0]
    for i, (x_, y_, pm) in enumerate(zip(x, y, pred_map)):
        ax = plt.subplot(nrows,ncols,i+1)
        ax.imshow(convert_image_tensor_to_ndarray(x_))
        for l in locs:
            xmin = int(l[1] - step_size/2)
            ymin = int(l[0] - step_size/2)

            iscorrect = pm[l[0], l[1]] == y_
            # print(xmin, ymin, iscorrect)
            color = 'green' if iscorrect else 'red'

            rect = patches.Rectangle((xmin, ymin), step_size, step_size, linewidth=0.5, edgecolor=color, facecolor=color, alpha=0.25)
            # rect = patches.Rectangle((l[1],l[0]), 2*step_size, 2*step_size, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        # break
    plt.savefig(f'many_fixations_img_eps={args.eps}.png')
    
    

    
        


# imgs = np.concatenate(imgs, 0)
# labels = np.concatenate(labels, 0)
# logit_maps = np.concatenate(logit_maps, 0)
# ofn = f'{os.path.dirname(os.path.dirname(args.ckp))}/many_fixations_results_eps={args.eps:.4f}.npz'
# np.savez(ofn, imgs=imgs, labels=labels, logit_maps=logit_maps)