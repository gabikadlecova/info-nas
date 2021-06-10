import numpy as np
import os

import torch

from info_nas.datasets.config import cfg
from info_nas.datasets.io.create_dataset import load_io_dataset, dataset_from_pretrained
from arch2vec.extensions.get_nasbench101_model import get_nasbench_datasets
from arch2vec.preprocessing.gen_json import gen_json_file

from info_nas.datasets.networks.pretrained import pretrain_network_dataset
from nasbench_pytorch.datasets.cifar10 import prepare_dataset


def get_labeled_unlabeled_datasets(nasbench, nb_dataset='../data/nb_dataset.json',
                                   dataset='../data/cifar/', seed=1, percent_labeled=0.01,
                                   train_labeled_path='../data/train_labeled.pt',
                                   valid_labeled_path='../data/valid_labeled.pt',
                                   train_pretrained='../data/train_checkpoints/',
                                   valid_pretrained='../data/valid_checkpoint/',
                                   device=None, config=None):
    # creates/loads both the original dataset and the labeled io dataset

    if config is None:
        config = cfg

    nb_dataset = _generate_or_load_nb_dataset(nasbench, save_path=nb_dataset, seed=seed, **config['nb_dataset'])

    if isinstance(dataset, str):
        dataset = prepare_dataset(root=dataset, random_state=seed, **config['cifar-10'])

    train_hashes, valid_hashes = split_to_labeled(nb_dataset, seed=seed, percent_labeled=percent_labeled)

    # networks from train set
    train_labeled = _create_or_load_labeled(nasbench, dataset, train_pretrained, train_labeled_path, train_hashes,
                                            seed=seed, device=device, config=config)

    # networks from valid set
    valid_labeled = _create_or_load_labeled(nasbench, dataset, valid_pretrained, valid_labeled_path, valid_hashes,
                                            seed=seed, device=device, config=config)

    # arch2vec already performed some preprocessing (e.g. padding of smaller adjacency matrices)
    train_data = _ops_adj_from_hashes(train_labeled[0], nb_dataset["train"])
    valid_data = _ops_adj_from_hashes(valid_labeled[0], nb_dataset["val"])

    labeled_dataset = {
        "train_io": train_labeled,
        "train_net": train_data,
        "valid_io": valid_labeled,
        "valid_net": valid_data
    }

    return labeled_dataset, nb_dataset


def split_to_labeled(dataset, seed=1, percent_labeled=0.01):
    state = np.random.RandomState(seed) if seed is not None else np.random

    train, valid = dataset["train"], dataset["val"]
    train_hashes, valid_hashes = _get_hashes(train[0]), _get_hashes(valid[0])

    train_hashes = state.choice(train_hashes, int(percent_labeled) * len(train_hashes), replace=False)
    valid_hashes = state.choice(valid_hashes, int(percent_labeled) * len(valid_hashes), replace=False)

    return train_hashes, valid_hashes


def _check_hashes(hashes, reference):
    for h in hashes:
        if h not in reference:
            raise ValueError(f"Hash {h} missing in the reference dataset (possible cause could be a different seed"
                             f" used for dataset splits).")


def _load_labeled(net_dir, reference_hashes, batched=False, device=None):
    hashes, inputs, outputs = load_io_dataset(net_dir, device=device)

    if batched:
        reference_hashes = _get_hashes(reference_hashes)

    _check_hashes(hashes, reference_hashes)

    return hashes, inputs, outputs


def _get_hashes(batched_list):
    return [h for batch in batched_list for h in batch]


def _generate_or_load_nb_dataset(nasbench, save_path=None, seed=1, **kwargs):
    if save_path is not None and os.path.exists(save_path):
        dataset = save_path
    else:
        dataset = gen_json_file(nasbench=nasbench, save_path=save_path)

    return get_nasbench_datasets(dataset, seed=seed, **kwargs)


def _pretrain_if_needed(pretrained_path, nasbench, dataset, net_hashes, device=None, **kwargs):
    if not os.path.exists(pretrained_path):
        os.mkdir(pretrained_path)

    if not len(os.listdir(pretrained_path)):
        pretrain_network_dataset(net_hashes, nasbench, dataset, device=device, dir_path=pretrained_path,
                                 **kwargs)


def _create_or_load_labeled(nasbench, dataset, pretrained_path, labeled_path, hashes, seed=1, device=None, config=None):
    if config is None:
        config = cfg

    if os.path.exists(labeled_path):
        labeled = _load_labeled(labeled_path, hashes, device=device)
    else:
        _pretrain_if_needed(pretrained_path, nasbench, dataset, hashes, device=device, **config['pretrain'])
        labeled = dataset_from_pretrained(pretrained_path, nasbench, dataset, labeled_path,
                                          random_state=seed, device=device, **config['io'])

    return labeled


def _ops_adj_from_hashes(net_hashes, nb_dataset):
    net_dict = {}

    # find data in batches
    for batch in nb_dataset:
        for item in batch:
            if item[0] in net_hashes:
                net_dict[item[0]] = item[1:]

    # return dataset in the same order as hashes
    ops = []
    adj = []
    for h in net_hashes:
        data = net_dict[h]

        adj.append(data[1])
        ops.append(data[2])

    return torch.stack(adj), torch.stack(ops)
