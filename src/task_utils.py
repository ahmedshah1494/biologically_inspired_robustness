from mllib.datasets.dataset_factory import (ImageDatasetFactory,
                                            SupportedDatasets)
import torchvision
from adversarialML.biologically_inspired_models.src.trainers import AdversarialParams
from adversarialML.biologically_inspired_models.src.utils import gethostname
from mllib.adversarial.attacks import (AttackParamFactory, SupportedAttacks,
                                       SupportedBackend)
from mllib.runners.configs import BaseExperimentConfig

hostname = gethostname()
if 'bridges2' in hostname:
    logdir_root = '/ocean/projects/cis220031p/mshah1'
else:
    logdir_root = '/share/workhorse3/mshah1'

LOGDIR = f'{logdir_root}/biologically_inspired_models/logs/'

def get_cifar10_params(num_train=25000, num_test=1000):
    p = ImageDatasetFactory.get_params()
    p.dataset = SupportedDatasets.CIFAR10
    p.datafolder = logdir_root
    p.max_num_train = num_train
    p.max_num_test = num_test
    return p

def get_tiny_imagenet_params(num_train=100_000, num_test=10_000):
    p = ImageDatasetFactory.get_params()
    p.dataset = SupportedDatasets.TINY_IMAGENET
    p.datafolder = f'{logdir_root}/tiny-imagenet-200/bin/'
    p.max_num_train = num_train
    p.max_num_test = num_test
    return p

def get_random_crop_flip_transforms(size, padding):
    return [
        torchvision.transforms.RandomCrop(size, padding=padding, padding_mode='reflect'),
        torchvision.transforms.RandomHorizontalFlip()
    ]

def get_resize_crop_flip_transforms(size, padding):
    return [
        torchvision.transforms.Resize(size),
        *(get_random_crop_flip_transforms(size, padding))
    ]

def get_resize_crop_flip_autoaugment_transforms(size, padding, profile):
    return [
        *(get_resize_crop_flip_transforms(size, padding)),
        torchvision.transforms.AutoAugment(profile)
    ]

def get_dataset_params(datafolder, dataset, num_train=13_000, num_test=500, train_transforms=None, test_transforms=None):
    p = ImageDatasetFactory.get_params()
    p.dataset = dataset
    p.datafolder = datafolder
    p.max_num_train = num_train
    p.max_num_test = num_test
    if train_transforms is not None:
        train_transforms.append(torchvision.transforms.ToTensor())
    else:
        train_transforms = [torchvision.transforms.ToTensor()]
    if test_transforms is not None:
        test_transforms.append(torchvision.transforms.ToTensor())
    else:
        test_transforms = [torchvision.transforms.ToTensor()]
    p.custom_transforms = (
        torchvision.transforms.Compose(train_transforms) if len(train_transforms) > 1 else train_transforms[0],
        torchvision.transforms.Compose(test_transforms) if len(test_transforms) > 1 else test_transforms[0]
    )
    return p

def get_imagenet_params(num_train=float('inf'), num_test=float('inf')):
    p = ImageDatasetFactory.get_params()
    p.dataset = SupportedDatasets.IMAGENET
    p.datafolder = f'{logdir_root}/imagenet/shards2/'
    p.max_num_train = num_train
    p.max_num_test = num_test
    return p

def set_common_training_params(p: BaseExperimentConfig):
    p.batch_size = 256
    p.trainer_params.training_params.nepochs = 200
    p.num_trainings = 10
    p.logdir = LOGDIR
    p.trainer_params.training_params.early_stop_patience = 20
    p.trainer_params.training_params.tracked_metric = 'val_loss'
    p.trainer_params.training_params.tracking_mode = 'min'

def set_adv_params(p: AdversarialParams, test_eps):
    p.training_attack_params = None
    def eps_to_attack(eps):
        atk_p = AttackParamFactory.get_attack_params(SupportedAttacks.APGDLINF, SupportedBackend.TORCHATTACKS)
        atk_p.eps = eps
        atk_p.nsteps = 50
        atk_p.step_size = eps/40
        atk_p.random_start = True
        return atk_p
    p.testing_attack_params = [eps_to_attack(eps) for eps in test_eps]

def get_imagenet10_params(num_train=13_000, num_test=500, train_transforms=None, test_transforms=None):
    return get_dataset_params(f'{logdir_root}/imagenet-100/bin/64', SupportedDatasets.IMAGENET10, 
                                num_train, num_test, train_transforms, test_transforms)

def get_imagenet100_params(num_train=25, num_test=1, train_transforms=None, test_transforms=None):
    return get_dataset_params(f'{logdir_root}/imagenet-100/bin/64', SupportedDatasets.IMAGENET100, 
                                num_train, num_test, train_transforms, test_transforms)

def get_imagenet100_64_params(num_train=127500, num_test=1000, train_transforms=None, test_transforms=None):
    return get_dataset_params(f'{logdir_root}/imagenet-100/bin/64', SupportedDatasets.IMAGENET100_64, 
                                num_train, num_test, train_transforms, test_transforms)

def get_imagenet75_64_params(num_train=127500, num_test=1000, train_transforms=None, test_transforms=None):
    return get_dataset_params(f'{logdir_root}/imagenet-75/bin/64', SupportedDatasets.IMAGENET75_64, 
                                num_train, num_test, train_transforms, test_transforms)

def get_imagenet100_81_params(num_train=127500, num_test=5000, train_transforms=None, test_transforms=None):
    return get_dataset_params(f'{logdir_root}/imagenet-100/bin/81', SupportedDatasets.IMAGENET100_81, 
                                num_train, num_test, train_transforms, test_transforms)