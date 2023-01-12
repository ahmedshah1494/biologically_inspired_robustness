from argparse import ArgumentParser
from importlib import import_module
import torch
import numpy as np
from adversarialML.biologically_inspired_models.src.runners import load_params_into_model
from adversarialML.biologically_inspired_models.src.utils import load_pickle
from matplotlib import pyplot as plt
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

parser = ArgumentParser()
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--ckp', type=str)
args = parser.parse_args()

task = get_task_class_from_str(args.task)()
model_params = task.get_model_params()
model_params.preprocessing_params.layer_params[1].loc_mode = 'center'
model_params.preprocessing_params.layer_params[1].view_scale = None
model = model_params.cls(model_params)

ckp = torch.load(args.ckp)
model = load_params_into_model(ckp, model)
model = model.eval()

x = torch.rand(1,3,224,224, requires_grad=True)

# ds_params = task.get_dataset_params()
# _, _, test_dataset, _ = ds_params.cls.get_image_dataset(ds_params)
test_dataset = np.load('/home/mshah1/workhorse3/ecoset-10/bin/320/test.pkl.npz')['X']

nimgs = 5
idxs = np.random.choice(len(test_dataset), nimgs, replace=False)
x = torch.stack([torch.FloatTensor(test_dataset[i]).permute(2,0,1)/255. for i in idxs], 0)
x.requires_grad = True
# model(x)
logits, r = model.forward_and_reconstruct(x)
px = model.preprocess(x)
if isinstance(px, tuple):
    px = px[0]
recon = r
model.compute_loss(x, torch.randint(0,10, (nimgs,)), return_logits=False).backward()
print(torch.norm(x.grad.reshape(-1)))
ncol = 4
for i in range(nimgs):
    print(torch.norm(x[i] - px[i], p=2, dim=[0,1,2]))
    plt.subplot(nimgs , ncol, i*ncol + 1)
    plt.imshow(convert_image_tensor_to_ndarray(torch.clamp(x[i],0.,1.).float()))
    plt.xticks([],[])
    plt.yticks([],[])
    plt.subplot(nimgs , ncol, i*ncol + 2)    
    plt.xticks([],[])
    plt.yticks([],[])
    plt.imshow(convert_image_tensor_to_ndarray(torch.clamp(px[i],0.,1.).float()))
    plt.subplot(nimgs , ncol, i*ncol + 3)
    plt.imshow(convert_image_tensor_to_ndarray(torch.clamp(recon[i],0.,1.).float()))
    plt.xticks([],[])
    plt.yticks([],[])
    plt.subplot(nimgs , ncol, i*ncol + 4)
    residual = (x[i] - recon[i]).abs()
    residual = residual + (1-residual.max())
    plt.imshow(convert_image_tensor_to_ndarray(residual))
    plt.xticks([],[])
    plt.yticks([],[])
plt.tight_layout()
plt.savefig('recon.png')