from __future__ import absolute_import
import warnings

from .cuhk01 import CUHK01
from .cuhk03 import CUHK03
from .dukemtmc import DukeMTMC
from .market1501 import Market1501
from .market1501_2 import Market1501_2
from .viper import VIPeR

import os
from .data import transforms as T
from .data.preprocessor import Preprocessor
from .data.sampler import RandomIdentitySampler
from torch.utils.data import DataLoader


__factory = {
    'viper': VIPeR,
    'cuhk01': CUHK01,
    'cuhk03': CUHK03,
    'market1501': Market1501,
    'market1501_2':Market1501_2,
    'dukemtmc': DukeMTMC,
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'viper', 'cuhk01', 'cuhk03',
        'market1501', and 'dukemtmc'.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)



def get_data(name, split_id, data_dir, height, width, batch_size, num_instances,
            workers, combine_trainval, sample_instance=False):
    root = os.path.join(data_dir, name)

    dataset = create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing()
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])
    if sample_instance:
        print("sample person and per person images")
        sampler = RandomIdentitySampler(train_set, num_instances)
        shuffle = False
    else:
        print("sample person as cls task")
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler = sampler,
        shuffle = shuffle, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, val_loader, test_loader
