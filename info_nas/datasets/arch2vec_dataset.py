import math

import numpy as np
import os

import torch

from info_nas.config import local_dataset_cfg
from info_nas.datasets.io.create_dataset import load_io_dataset, dataset_from_pretrained
from arch2vec.extensions.get_nasbench101_model import get_nasbench_datasets
from arch2vec.preprocessing.gen_json import gen_json_file
from nasbench_pytorch.datasets.cifar10 import prepare_dataset


def _split_pretrain_paths(paths: str):
    return paths.split(',')


def get_labeled_unlabeled_datasets(nasbench, nb_dataset='../data/nb_dataset.json',
                                   dataset='../data/cifar/', seed=1,
                                   train_labeled_path='../data/train_labeled.pt',
                                   valid_labeled_path='../data/valid_labeled.pt',
                                   train_pretrained='../data/train_checkpoints/',
                                   valid_pretrained='../data/valid_checkpoints/',
                                   device=None, config=None):
    # creates/loads both the original dataset and the labeled io dataset

    if config is None:
        config = local_dataset_cfg

    train_pretrained = _split_pretrain_paths(train_pretrained)
    valid_pretrained = _split_pretrain_paths(valid_pretrained)

    nb_dataset = generate_or_load_nb_dataset(nasbench, save_path=nb_dataset, seed=seed, batch_size=None,
                                             **config['nb_dataset'])

    print('Processing labeled nets for the training set...')
    # networks from train set
    train_labeled = _create_or_load_labeled(nasbench, dataset, train_pretrained, train_labeled_path,
                                            seed=seed, device=device, config=config)

    print('Processing labeled nets for the validation set...')
    # networks from valid set
    valid_labeled = _create_or_load_labeled(nasbench, dataset, valid_pretrained, valid_labeled_path,
                                            seed=seed, device=device, config=config)

    # arch2vec already performed some preprocessing (e.g. padding of smaller adjacency matrices)
    _ops_adj_to_repo(train_labeled, nb_dataset["train"])
    _ops_adj_to_repo(valid_labeled, nb_dataset["val"])

    labeled_dataset = {
        "train": train_labeled,
        "valid": valid_labeled
    }

    return labeled_dataset, nb_dataset


def split_to_labeled(dataset, seed=1, percent_labeled=0.01):
    state = np.random.RandomState(seed) if seed is not None else np.random

    train, valid = dataset["train"], dataset["val"]
    train_hashes, valid_hashes = train[0], valid[0]

    train_hashes_chosen = state.choice(train_hashes, math.ceil(percent_labeled * len(train_hashes)), replace=False)
    valid_hashes_chosen = state.choice(valid_hashes, math.ceil(percent_labeled * len(valid_hashes)), replace=False)

    print(f"Split the dataset (percent labeled = {percent_labeled}) - "
          f"{len(train_hashes_chosen)}/{len(train_hashes)} labeled networks chosen from the train set, "
          f"{len(valid_hashes_chosen)}/{len(valid_hashes)} labeled networks chosen from the validation set.")
    return train_hashes_chosen, valid_hashes_chosen


# TODO tohle na odebrání z val setu potentially
def _check_hashes(hashes, reference):
    for h in hashes:
        if h not in reference:
            raise ValueError(f"Hash {h} missing in the reference dataset (possible cause could be a different seed"
                             f" used for dataset splits).")


def generate_or_load_nb_dataset(nasbench, save_path=None, seed=1, batch_size=None, val_batch_size=None, **kwargs):
    if save_path is not None and os.path.exists(save_path):
        print(f"Loading nasbench dataset (arch2vec) from {save_path}")
        dataset = save_path
    else:
        print(f"Generating nasbench dataset (arch2vec){'.' if save_path is None else f', save path = {save_path}.'}")
        dataset = gen_json_file(nasbench=nasbench, save_path=save_path)

    return get_nasbench_datasets(dataset, batch_size=batch_size, val_batch_size=None, seed=seed, **kwargs)


def _check_pretrained(pretrained_path):
    err_loading = f"No pretrained networks found in the specified path - {pretrained_path}."

    if not os.path.exists(pretrained_path):
        raise ValueError(err_loading)

    if not len(os.listdir(pretrained_path)):
        raise ValueError(err_loading)


def _create_or_load_labeled(nasbench, dataset, pretrained_paths, labeled_path, seed=1, device=None, config=None):
    if config is None:
        config = local_dataset_cfg

    if os.path.exists(labeled_path):
        print(f'Loading labeled dataset from {labeled_path}.')
        labeled = load_io_dataset(labeled_path)
    else:
        if isinstance(dataset, str):
            dataset = prepare_dataset(root=dataset, random_state=seed, **config['cifar-10'])

        # check all folders for pretrain files
        for path in pretrained_paths:
            _check_pretrained(path)

        print(f'Creating labeled dataset from pretrained networks (saving to {labeled_path}).')
        labeled = dataset_from_pretrained(pretrained_paths, nasbench, dataset, labeled_path,
                                          random_state=seed, device=device, **config['io'])

    return labeled


def _ops_adj_to_repo(dataset, nb_dataset):
    net_repo = dataset['net_repo']

    # find corresponding network graph in nb dataset
    for i_batch, item in enumerate(nb_dataset[0]):
        if item in net_repo:
            net_repo[item]['adj'] = nb_dataset[1][i_batch]
            net_repo[item]['ops'] = nb_dataset[2][i_batch]
