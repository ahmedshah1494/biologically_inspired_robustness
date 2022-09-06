import os
import torch
import torchvision
import numpy as np
import json
from tqdm import tqdm
from multiprocessing import Pool

DATASET = 'imagenet-100'
RESIZE = 64

outfolder = f'/home/mshah1/workhorse3/{DATASET}/bin/'
if RESIZE > 0:
    outfolder = os.path.join(outfolder, str(RESIZE))
datafolder = f'/home/mshah1/workhorse3/{DATASET}/'

if RESIZE > 0:
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(RESIZE),
        torchvision.transforms.CenterCrop(RESIZE),
    ])
else:
    transforms = None

# if DATASET.startswith('tiny-imagenet'):
#     transforms = None
# elif DATASET == 'imagenet-10':
#     transforms = torchvision.transforms.Compose([
#         torchvision.transforms.Resize(320),
#         torchvision.transforms.CenterCrop(320),
#     ])
# elif DATASET == 'imagenet-100':
#     transforms = torchvision.transforms.Compose([
#         torchvision.transforms.Resize(320),
#         torchvision.transforms.CenterCrop(320),
#     ])

if not os.path.exists(outfolder):
    os.makedirs(outfolder)

print('creating datasets...')
train_dataset = torchvision.datasets.ImageFolder(os.path.join(datafolder, 'train'), transform=transforms)
test_dataset = torchvision.datasets.ImageFolder(os.path.join(datafolder, 'val'), transform=transforms)
assert test_dataset.class_to_idx == train_dataset.class_to_idx

print('creating idx2word')
if os.path.exists(os.path.join(datafolder, 'words.txt')) and os.path.exists(os.path.join(datafolder, 'wnids.txt')):
    wnid2word = np.loadtxt(os.path.join(datafolder, 'words.txt'), dtype=str, delimiter='\t')
    wnids = np.loadtxt(os.path.join(datafolder, 'wnids.txt'), dtype=str, delimiter='\t')
    wnid2word = {i: n for i,n in zip(wnid2word[:, 0], wnid2word[:,1]) if i in wnids}
    idx2word = {test_dataset.class_to_idx[wi]: w for wi,w in wnid2word.items()}
    with open(os.path.join(outfolder, 'idx2word.json'), 'w') as f:
        json.dump(idx2word, f)

def save_dataset(dataset, outfile):
    # for x, y in tqdm.tq(dataset):
    #     xa = np.array(x)
    #     X.append(xa)
    #     Y.append(y)
    with Pool(8) as p:
        data = p.map(dataset.__getitem__, range(len(dataset)))
    X, Y = zip(*data)
    X = [np.array(x) for x in X]
    X = np.stack(X, 0)
    Y = np.array(Y)
    print(X.shape, Y.shape)
    np.savez_compressed(os.path.join(outfolder, outfile), X=X, Y=Y)

print('saving test dataset...')
save_dataset(test_dataset, 'test.pkl')
print('saving train dataset...')
save_dataset(train_dataset, 'train.pkl')
    