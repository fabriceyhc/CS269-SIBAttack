import numpy as np
import torch
import torchvision
from torchvision import transforms

from RandAugment import RandAugment
from transformations.image.mixtures import mixup2
from transformations.image.mixtures import cutmix2
from transformations.image.mixtures import tile
from transformations.image.mixtures.utils import *


def get_loader(config):

    train_transform = torchvision.transforms.Compose([
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    if config['use_basicaug']:
        test_transform.transforms.insert(0, transforms.RandomHorizontalFlip())
        test_transform.transforms.insert(0, transforms.RandomCrop(32, padding=4))

    if config['use_randaug']:
        n = config['randaug_n']
        m = config['randaug_m']
        test_transform.transforms.insert(0, RandAugment(n=2, m=3))

    if config['dataset'] == 'CIFAR10':
        _MEAN, _STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        test_transform.transforms.insert(
            len(test_transform.transforms), transforms.Normalize(_MEAN, _STD))
        train_dataset = torchvision.datasets.CIFAR10(
            dataset_dir, train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            dataset_dir, train=False, transform=test_transform, download=True)
    elif config['dataset'] == 'MNIST':
        _MEAN, _STD = (0.1307,), (0.3081,)
        train_dataset = torchvision.datasets.MNIST(
            dataset_dir, train=True, transform=train_transform, download=True)
        test_transform.transforms.insert(
            len(test_transform.transforms), transforms.Normalize(_MEAN, _STD))
        test_dataset = torchvision.datasets.MNIST(
            dataset_dir, train=False, transform=test_transform, download=True)
    else:
        raise InputError("dataset not supported...")

    collator = torch.utils.data.dataloader.default_collate
    
    if config['use_mixup2']:
        collator = mixup2.Mixup2Collator(
            alpha=config['alpha'],  
            target_pairs=config['target_pairs'], 
            target_prob=config['target_prob'], 
            num_classes=config['n_classes']
        )
    if config['use_cutmix2']:
        collator = cutmix2.CutMix2Collator(
            alpha=config['alpha'],  
            target_pairs=config['target_pairs'], 
            target_prob=config['target_prob'], 
            resize_prob=config['resize_prob'], 
            num_classes=config['n_classes']
        )
    if config['use_tile']:
        collator = tile.TileCollator(
            num_tiles=config['num_tiles'],  
            target_pairs=config['target_pairs'], 
            target_prob=config['target_prob'], 
            num_classes=config['n_classes']
        )

    target_transform = lambda x : x
    if config['use_reducemix']:
        target_transform = ReduceMix(
            num_clases=config['n_classes'],
            return_tensors='pt'
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collator,
        num_workers=config['num_workers'],
        target_transform=target_transform,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader

class ReduceMix:
    def __init__(self, num_classes=2, return_tensors='pt'):
        self.num_classes = num_classes
        self.return_tensors = return_tensors

    def __call__(self, targets):
        if len(targets.shape) >= 2:
            targets = [np.argmax(y) if i == 1 else self.num_classes - 1 for (i, y) in zip(np.count_nonzero(targets, axis=-1), targets)]
        if self.return_tensors == 'pt':
            targets = torch.tensor(targets).long()
        if self.return_tensors == 'np':
            targets = np.array(targets)
        return targets