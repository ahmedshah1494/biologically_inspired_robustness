from collections import OrderedDict
import os
from pathlib import Path
import shutil

from imageio.v3 import imread, imwrite
from PIL import Image
import pysaliency
from pysaliency.baseline_utils import BaselineModel, CrossvalidatedBaselineModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

from tqdm import tqdm


from deepgaze_pytorch.layers import (
    Conv2dMultiInput,
    LayerNorm,
    LayerNormMultiInput,
    Bias,
    FlexibleScanpathHistoryEncoding
)

from deepgaze_pytorch.modules import DeepGazeIII, DeepGazeII, FeatureExtractor
from deepgaze_pytorch.data import ImageDataset, ImageDatasetSampler, FixationDataset, FixationMaskTransform
from deepgaze_pytorch.training import _train

from argparse import ArgumentParser
from importlib import import_module
from rblur.runners import load_params_into_model

parser = ArgumentParser()
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--ckp', type=str)
parser.add_argument('--layers', nargs='+', type=str)
parser.add_argument('--expname', type=str, default='')
parser.add_argument('--stage', type=str, choices=['pretrain', 'fine-tune', 'fine-tune-clickme'])
parser.add_argument('--arch', type=str, choices=['deepgaze2', 'deepgaze3'])
args = parser.parse_args()

def get_task_class_from_str(s):
    split = s.split('.')
    modstr = '.'.join(split[:-1])
    cls_name =  split[-1]
    mod = import_module(modstr)
    task_cls = getattr(mod, cls_name)
    return task_cls

def build_saliency_network(input_channels):
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNorm(input_channels)),
        ('conv0', nn.Conv2d(input_channels, 8, (1, 1), bias=False)),
        ('bias0', Bias(8)),
        ('softplus0', nn.Softplus()),

        ('layernorm1', LayerNorm(8)),
        ('conv1', nn.Conv2d(8, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),

        ('layernorm2', LayerNorm(16)),
        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
        ('bias2', Bias(1)),
        ('softplus2', nn.Softplus()),
    ]))


def build_scanpath_network():
    return nn.Sequential(OrderedDict([
        ('encoding0', FlexibleScanpathHistoryEncoding(in_fixations=4, channels_per_fixation=3, out_channels=128, kernel_size=[1, 1], bias=True)),
        ('softplus0', nn.Softplus()),

        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
    ]))


def build_fixation_selection_network(scanpath_features=16):
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNormMultiInput([1, scanpath_features])),
        ('conv0', Conv2dMultiInput([1, scanpath_features], 128, (1, 1), bias=False)),
        ('bias0', Bias(128)),
        ('softplus0', nn.Softplus()),

        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),

        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
    ]))

def prepare_spatial_dataset(stimuli, fixations, centerbias, batch_size, path=None):
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        lmdb_path = str(path)
    else:
        lmdb_path = None

    dataset = ImageDataset(
        stimuli=stimuli,
        fixations=fixations,
        centerbias_model=centerbias,
        transform=FixationMaskTransform(sparse=False),
        average='image',
        lmdb_path=lmdb_path,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=ImageDatasetSampler(dataset, batch_size=batch_size),
        pin_memory=False,
        num_workers=0,
    )

    return loader

def prepare_scanpath_dataset(stimuli, fixations, centerbias, batch_size, path=None):
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        lmdb_path = str(path)
    else:
        lmdb_path = None

    dataset = FixationDataset(
        stimuli=stimuli,
        fixations=fixations,
        centerbias_model=centerbias,
        included_fixations=[-1, -2, -3, -4],
        allow_missing_fixations=True,
        transform=FixationMaskTransform(sparse=False),
        average='image',
        lmdb_path=lmdb_path,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=ImageDatasetSampler(dataset, batch_size=batch_size),
        pin_memory=False,
        num_workers=0,
    )

    return loader

def convert_stimulus(input_image):
    size = input_image.shape[0], input_image.shape[1]
    if size[0] < size[1]:
        new_size = 768, 1024
    else:
        new_size = 1024,768
    
    # pillow uses width, height
    new_size = tuple(list(new_size)[::-1])
    
    new_stimulus = np.array(Image.fromarray(input_image).resize(new_size, Image.BILINEAR))
    return new_stimulus

def convert_fixations(stimuli, fixations):
    new_fixations = fixations.copy()
    for n in tqdm(list(range(len(stimuli)))):
        stimulus = stimuli.stimuli[n]
        size = stimulus.shape[0], stimulus.shape[1]
        if size[0] < size[1]:
            new_size = 768, 1024
        else:
            new_size = 1024,768
        x_factor = new_size[1] / size[1]
        y_factor = new_size[0] / size[0]
        
        inds = new_fixations.n == n
        new_fixations.x[inds] *= x_factor
        new_fixations.y[inds] *= y_factor
        new_fixations.x_hist[inds] *= x_factor
        new_fixations.y_hist[inds] *= y_factor
    
    return new_fixations

def convert_fixation_trains(stimuli, fixations):
    train_xs = fixations.train_xs.copy()
    train_ys = fixations.train_ys.copy()
    
    for i in tqdm(range(len(train_xs))):
        n = fixations.train_ns[i]
        
        size = stimuli.shapes[n][0], stimuli.shapes[n][1]
        
        if size[0] < size[1]:
            new_size = 768, 1024
        else:
            new_size = 1024,768
        
        x_factor = new_size[1] / size[1]
        y_factor = new_size[0] / size[0]
        
        train_xs[i] *= x_factor
        train_ys[i] *= y_factor
        
    new_fixations = pysaliency.FixationTrains(
        train_xs = train_xs,
        train_ys = train_ys,
        train_ts = fixations.train_ts.copy(),
        train_ns = fixations.train_ns.copy(),
        train_subjects = fixations.train_subjects.copy(),
        attributes={key: getattr(fixations, key).copy() for key in fixations.__attributes__ if key not in ['subjects', 'scanpath_index']},
    )
    return new_fixations



def convert_stimuli(stimuli, new_location: Path):
    assert isinstance(stimuli, pysaliency.FileStimuli)
    new_stimuli_location = new_location / 'stimuli'
    new_stimuli_location.mkdir(parents=True, exist_ok=True)
    new_filenames = []
    for filename in tqdm(stimuli.filenames):
        stimulus = imread(filename)
        new_stimulus = convert_stimulus(stimulus)
        
        basename = os.path.basename(filename)
        new_filename = new_stimuli_location / basename
        if new_stimulus.size != stimulus.size:
            imwrite(new_filename, new_stimulus)
        else:
            #print("Keeping")
            shutil.copy(filename, new_filename)
        new_filenames.append(new_filename)
    return pysaliency.FileStimuli(new_filenames)

dataset_directory = Path('/home/mshah1/workhorse3/pysaliency_datasets')
train_directory = Path(f'/home/mshah1/workhorse3/train_{args.arch}')

backbone_task = get_task_class_from_str(args.task)()
model_params = backbone_task.get_model_params()
# model_params.feature_model_params.layer_params[1].input_shape = [3, 300, 300]
backbone = model_params.cls(model_params)

source_sd = torch.load(args.ckp)
backbone = load_params_into_model(source_sd, backbone)

class ToFloatImage(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x.float()
        if x.max() > 1:
            x = x / 255
        sf = 224 / min(x.shape[2:])        
        # scale and resize image and center bias to match training regime
        if sf != 1:
            x = torch.nn.functional.interpolate(x, scale_factor=sf)
        return x
    
backbone = nn.Sequential(
    ToFloatImage(),
    backbone
)

device = 'cuda'

feature_extractor = FeatureExtractor(backbone, args.layers)
x = torch.rand(1,3,224,224)
feats = feature_extractor(x)
featdim = sum([f.shape[1] for f in feats])

if args.arch == 'deepgaze3':
    model = DeepGazeIII(
        features=feature_extractor,
        saliency_network=build_saliency_network(featdim),
        scanpath_network=None,
        fixation_selection_network=build_fixation_selection_network(scanpath_features=0),
        downsample=1.,
        readout_factor=4,
        saliency_map_factor=4,
        included_fixations=[],
    )
elif args.arch == 'deepgaze2':
    model = DeepGazeII(
    features=FeatureExtractor(backbone, args.layers),
    readout_network=nn.Conv2d(featdim, 1, 1)
)
print(model)


if args.stage == 'pretrain':
    SALICON_train_stimuli, SALICON_train_fixations = pysaliency.get_SALICON_train(location=dataset_directory)
    SALICON_val_stimuli, SALICON_val_fixations = pysaliency.get_SALICON_val(location=dataset_directory)

    # parameters taken from an early fit for MIT1003. Since SALICON has many more fixations, the bandwidth won't be too small
    SALICON_centerbias = BaselineModel(stimuli=SALICON_train_stimuli, fixations=SALICON_train_fixations, bandwidth=0.0217, eps=2e-13, caching=False)

    # takes quite some time, feel free to set to zero
    train_baseline_log_likelihood = 0.46408017115279737 # SALICON_centerbias.information_gain(SALICON_train_stimuli, SALICON_train_fixations, verbose=True, average='image')
    val_baseline_log_likelihood = 0.4291592320821603 # SALICON_centerbias.information_gain(SALICON_val_stimuli, SALICON_val_fixations, verbose=True, average='image')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 60, 75, 90, 105, 120])

    train_loader = prepare_spatial_dataset(SALICON_train_stimuli, SALICON_train_fixations, SALICON_centerbias, batch_size=4, path=train_directory / 'lmdb_cache' / 'SALICON_train')
    validation_loader = prepare_spatial_dataset(SALICON_val_stimuli, SALICON_val_fixations, SALICON_centerbias, batch_size=4, path=train_directory / 'lmdb_cache' / 'SALICON_val')

    _train(train_directory / args.task.split('.')[-1] / args.expname / 'pretraining',
        model,
        train_loader, train_baseline_log_likelihood,
        validation_loader, val_baseline_log_likelihood,
        optimizer, lr_scheduler,
        minimum_learning_rate=1e-7,
        validation_epochs=5,
        device=device,
    )

elif args.stage == 'fine-tune':
    mit_stimuli_orig, mit_scanpaths_orig = pysaliency.external_datasets.get_mit1003_with_initial_fixation(location=dataset_directory)

    mit_scanpaths_twosize = convert_fixation_trains(mit_stimuli_orig, mit_scanpaths_orig)
    mit_stimuli_twosize = convert_stimuli(mit_stimuli_orig, train_directory / args.task.split('.')[-1] / args.expname / 'MIT1003_twosize')
    mit_fixations_twosize = mit_scanpaths_twosize[mit_scanpaths_twosize.lengths > 0]
    MIT1003_centerbias = CrossvalidatedBaselineModel(
        mit_stimuli_twosize,
        mit_fixations_twosize,
        bandwidth=10**-1.6667673342543432,
        eps=10**-14.884189168516073,
        caching=False,
    )

    crossval_fold = 0
    
    MIT1003_stimuli_train, MIT1003_fixations_train = pysaliency.dataset_config.train_split(mit_stimuli_twosize, mit_fixations_twosize, crossval_folds=10, fold_no=crossval_fold)
    MIT1003_stimuli_val, MIT1003_fixations_val = pysaliency.dataset_config.validation_split(mit_stimuli_twosize, mit_fixations_twosize, crossval_folds=10, fold_no=crossval_fold)

    train_baseline_log_likelihood = MIT1003_centerbias.information_gain(MIT1003_stimuli_train, MIT1003_fixations_train, verbose=True, average='image')
    val_baseline_log_likelihood = MIT1003_centerbias.information_gain(MIT1003_stimuli_val, MIT1003_fixations_val, verbose=True, average='image')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9, 12, 15, 18, 21, 24])

    train_loader = prepare_spatial_dataset(MIT1003_stimuli_train, MIT1003_fixations_train, MIT1003_centerbias, batch_size=4, path=train_directory / 'lmdb_cache' / f'MIT1003_train_spatial_{crossval_fold}')
    validation_loader = prepare_spatial_dataset(MIT1003_stimuli_val, MIT1003_fixations_val, MIT1003_centerbias, batch_size=4, path=train_directory / 'lmdb_cache' / f'MIT1003_val_spatial_{crossval_fold}')

    _train(train_directory / args.task.split('.')[-1] / args.expname / 'MIT1003_spatial' / f'crossval-10-{crossval_fold}',
        model,
        train_loader, train_baseline_log_likelihood,
        validation_loader, val_baseline_log_likelihood,
        optimizer, lr_scheduler,
        minimum_learning_rate=1e-7,
        device=device,
        startwith=train_directory / args.task.split('.')[-1] / args.expname / 'pretraining' / 'final.pth',
    )

elif args.stage == 'fine-tune-clickme':
    import torchvision
    import webdataset as wds
    import tqdm
    from scipy.special import softmax

    def identity(x):
        return x

    def toFloatTensor(x):
        return torch.FloatTensor(x)

    def transpose_to_chw(x):
        return x.transpose(2,1).transpose(1,0)

    def get_clickme_webdataset(root, dataset_name, centerbias=None, first_shard_idx=0, nshards=None, split='train', transform=None, shuffle=10_000, len_shard=10_000):
        if split =='train':
            urls = dataset_name+"-train-{}.tar"
        elif split =='val':
            urls = dataset_name+"-trainval-{}.tar"
        elif split == 'test':
            shuffle = 0
            urls = dataset_name+"-val-{}.tar"
        else:
            raise ValueError(f'split must be one of train, val, or test, but got {split}')
        if nshards > 1:
            shard_str = f'{{{first_shard_idx:06d}..{first_shard_idx+nshards-1:06d}}}'
        else:
            shard_str = f'{first_shard_idx:06d}'
        urls = os.path.join(root, urls.format(shard_str))
        print(nshards, urls)
        dataset = wds.WebDataset(urls, shardshuffle=(split=='train'), nodesplitter=wds.split_by_node)
        if split == 'train':
            dataset = dataset.shuffle(shuffle, initial=shuffle//2)
        dataset = dataset.decode("pil")
        dataset = dataset.compose(wds.filters.associate(lambda x: {'weight':1.}))
        if centerbias is not None:
            dataset = dataset.compose(wds.filters.associate(lambda x: {'centerbias':centerbias}))
            dataset = (
                dataset
        #         .to_tuple("jpg;png;jpeg cls heatmap.npy")
        #         .map_tuple(transform, identity, toFloatTensor)
                .map_dict(**({'jpg':transform, 'cls':identity, 'heatmap.npy': toFloatTensor, 'centerbias': toFloatTensor}))
                .map_dict(**({'jpg':identity, 'cls':identity, 'heatmap.npy': transpose_to_chw, 'centerbias': identity}))
                .rename(**({'image':'jpg', 'y':'cls', 'fixation_mask':'heatmap.npy', 'centerbias':'centerbias'}))        
                # .with_length(nshards*len_shard)
            )
        else:
            dataset = (
                dataset
        #         .to_tuple("jpg;png;jpeg cls heatmap.npy")
        #         .map_tuple(transform, identity, toFloatTensor)
                .map_dict(**({'jpg':transform, 'cls':identity, 'heatmap.npy': toFloatTensor}))
                .map_dict(**({'jpg':identity, 'cls':identity, 'heatmap.npy': transpose_to_chw}))
                .rename(**({'image':'jpg', 'y':'cls', 'fixation_mask':'heatmap.npy'}))        
                # .with_length(nshards*len_shard)
            )
        return dataset
    
    transforms=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224, max_size=225),
            torchvision.transforms.ToTensor()
        ])
    num_train_shards = 39
    num_test_shards = 5
    num_val_shards = 1
    clickme_train_dataset = get_clickme_webdataset('/home/mshah1/workhorse3/clickme/shards', 'clickme', nshards=num_train_shards, split='train', transform=transforms)
    # val_dataset = get_clickme_webdataset('/home/mshah1/workhorse3/clickme/shards', 'clickme', nshards=num_val_shards, split='val', transform=transforms)
    clickme_test_dataset = get_clickme_webdataset('/home/mshah1/workhorse3/clickme/shards', 'clickme', nshards=num_test_shards, split='test', transform=transforms)

    def get_clickme_images_and_fixations(dataset, N):
        clickme_images = []
        clickme_fixmaps = []
        def density_to_fixation(density_map):
        #     prob_map = softmax(density_map.reshape(density_map.shape[0], -1), 1).reshape(density_map.shape)
            prob_map = density_map / density_map.max()

            rng = np.random.Generator(np.random.PCG64(69239841))
            fixation_mask = rng.binomial(1, p=prob_map, size=prob_map.shape)
            return fixation_mask
        for batch in tqdm.tqdm(dataset.iterator()):
            x, m = batch['image'], batch['fixation_mask']
            m = m.squeeze(2).numpy()
            x = x.transpose(0,1).transpose(1,2).numpy()
            if x.max() <= 1:
                x = (x * 255).astype(int).astype(np.ubyte)
            m = density_to_fixation(m).astype(np.half)
            clickme_fixmaps.append(m)
            clickme_images.append(x)
            if len(clickme_images) >= N:
                break
        return clickme_images, clickme_fixmaps
    
    clickme_train_images_10K, clickme_train_fixmaps_10K = get_clickme_images_and_fixations(clickme_train_dataset, 10_000)
    clickme_val_images_10K, clickme_val_fixmaps_10K = get_clickme_images_and_fixations(clickme_test_dataset, 10_000)

    clickme_train_stimuli_10K = pysaliency.Stimuli(clickme_train_images_10K)
    clickme_val_stimuli_10K = pysaliency.Stimuli(clickme_val_images_10K)

    clickme_train_fixations_10K = pysaliency.Fixations.from_fixation_matrices([x.squeeze() for x in clickme_train_fixmaps_10K])
    clickme_val_fixations_10K = pysaliency.Fixations.from_fixation_matrices([x.squeeze() for x in clickme_val_fixmaps_10K])

    # parameters taken from an early fit for MIT1003. Since SALICON has many more fixations, the bandwidth won't be too small
    clickme_centerbias = BaselineModel(stimuli=clickme_train_stimuli_10K, fixations=clickme_train_fixations_10K, bandwidth=0.0217, eps=2e-13, caching=False)

    # takes quite some time, feel free to set to zero
    train_baseline_log_likelihood = 0.46408017115279737 # SALICON_centerbias.information_gain(SALICON_train_stimuli, SALICON_train_fixations, verbose=True, average='image')
    val_baseline_log_likelihood = 0.4291592320821603 # SALICON_centerbias.information_gain(SALICON_val_stimuli, SALICON_val_fixations, verbose=True, average='image')

    centerbias = clickme_centerbias._log_density(clickme_train_images_10K[0])
    transforms=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224, max_size=225),
            torchvision.transforms.ToTensor()
        ])
    num_train_shards = 39
    num_test_shards = 5
    num_val_shards = 1
    clickme_train_dataset = get_clickme_webdataset('/home/mshah1/workhorse3/clickme/shards', 'clickme', centerbias=centerbias, nshards=num_train_shards, split='train', transform=transforms)
    # val_dataset = get_clickme_webdataset('/home/mshah1/workhorse3/clickme/shards', 'clickme', nshards=num_val_shards, split='val', transform=transforms)
    clickme_test_dataset = get_clickme_webdataset('/home/mshah1/workhorse3/clickme/shards', 'clickme', centerbias=centerbias, nshards=num_test_shards, split='test', transform=transforms)

    num_workers = 1 # 8 // torch.cuda.device_count()
    BATCH_SIZE=64

    def density_to_fixation(density_map):
    #     prob_map = torch.softmax(density_map.reshape(density_map.shape[0], -1), 1).reshape(density_map.shape)
    #     prob_map[prob_map > 0] = 1.
        prob_map = density_map / density_map.max()
        prob_map = torch.bernoulli(prob_map)
        return prob_map
        
    def _collate_fn(batch):
        collated_batch = torch.utils.data.default_collate(batch)
        collated_batch.pop('__key__')
        collated_batch.pop('__url__')
        collated_batch.pop('y')
        for i in range(len(collated_batch['fixation_mask'])):
            collated_batch['fixation_mask'][i] = density_to_fixation(collated_batch['fixation_mask'][i])
        return collated_batch

    clickme_train_dataset = clickme_train_dataset.shuffle(10_000).batched(BATCH_SIZE, partial=False, collation_fn=_collate_fn)
    clickme_test_dataset = clickme_test_dataset.batched(BATCH_SIZE, partial=False, collation_fn=_collate_fn)

    clickme_train_loader = wds.WebLoader(clickme_train_dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=num_workers)
    clickme_val_loader = wds.WebLoader(clickme_test_dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=num_workers)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2, 3 ,4])

    _train(train_directory / args.task.split('.')[-1] / args.expname / 'clickme',
        model,
        clickme_train_loader, train_baseline_log_likelihood,
        clickme_val_loader, val_baseline_log_likelihood,
        optimizer, lr_scheduler,
        minimum_learning_rate=1e-7,
        device=device,
        startwith=train_directory / args.task.split('.')[-1] / args.expname / 'pretraining' / 'final.pth',
    )