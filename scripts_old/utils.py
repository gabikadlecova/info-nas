import os

import torch
import torchvision
from _old.datasets import prepare_labeled_dataset, split_off_valid
from _old.datasets.io.semi_dataset import labeled_network_dataset

from info_nas.io_dataset.transforms import ToTuple, SortByWeights, MultByWeights, IncludeBias


def mkdir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def experiment_transforms(cfg, use_accuracy=False):
    transforms = []
    transforms.append(IncludeBias())
    nr = cfg['scale'].get('normalize_row', False)
    transforms.append(MultByWeights(include_bias=True, normalize_row=nr))
    top_k = cfg['scale'].get('top_k', None)
    transforms.append(SortByWeights(return_top_n=top_k, after_sort_scale=None))
    if not use_accuracy:
        transforms.append(ToTuple())
    return torchvision.transforms.Compose(transforms)


def get_eval_set(data_name, dataset, nb, transforms, batch_size, config=None, split_ratio=None, use_larger_part=False):
    key = 'val' if data_name == 'valid' else data_name
    dataset, _ = prepare_labeled_dataset(dataset, nb, key=key, remove_labeled=False, config=config)
    if split_ratio is not None:
        larger_part, dataset = split_off_valid(dataset, ratio=split_ratio)
        dataset = larger_part if use_larger_part else dataset

    dataset = labeled_network_dataset(dataset, transforms=transforms)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=0)
