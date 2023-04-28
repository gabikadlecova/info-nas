import os
from abc import abstractmethod

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader


class BaseIOExtractor:
    @abstractmethod
    def get_io_data(self, net, data):
        pass


class BaseNetworkData:
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get_data(self, net_hash):
        pass

    @abstractmethod
    def get_hashes(self):
        pass


class IOHook:
    def __init__(self, save_inputs=False, save_outputs=True):
        self.outputs = []
        self.inputs = []
        self.save_inputs = save_inputs
        self.save_outputs = save_outputs

    def get_hook(self):
        def hook(_, i, o):
            ins = i[0].detach().cpu()
            outs = o.detach().cpu()
            self.inputs.append(ins)
            self.outputs.append(outs)

        return hook


def split_unlabeled_labeled(all_keys, labeled_keys):
    unlabeled_keys = [h for h in all_keys if h not in labeled_keys]
    return unlabeled_keys, labeled_keys


def apply_to_vals(dataset, func):
    if isinstance(dataset, dict):
        return {k: func(v) for k, v in dataset.items()}
    return [func(v) for v in dataset]


def datasets_to_iterable(data):
    if isinstance(data, dict) or isinstance(data, list):
        return data
    return [data]


class NetworkDataModule(pl.LightningDataModule):
    def __init__(self, network_data: BaseNetworkData, train_datasets, val_datasets=None, test_datasets=None,
                 batch_size=32):
        super().__init__()

        self.network_data = network_data
        self.batch_size = batch_size
        self.single_train = True

        self.data = {'train': datasets_to_iterable(train_datasets)}
        if val_datasets is not None:
            self.data['val'] = datasets_to_iterable(val_datasets) if val_datasets is not None else None
        if test_datasets is not None:
            self.data['test'] = datasets_to_iterable(test_datasets) if test_datasets is not None else None

    def prepare_data(self):
        for datasets in self.data.values():
            apply_to_vals(datasets, lambda d: d.load())

    def train_dataloader(self):
        loaders = self._get_loaders('train')
        return CombinedLoader(loaders, mode='max_size_cycle')

    def val_dataloader(self):
        return self._get_loaders('val')

    def test_dataloader(self):
        return self._get_loaders('test')

    def _get_loaders(self, key):
        if self.data[key] is None:
            return None

        shuffle = True if key == 'train' else False
        loaders = apply_to_vals(self.data[key], lambda d: DataLoader(d, batch_size=self.batch_size, shuffle=shuffle))
        return loaders if len(loaders) > 1 else loaders[0]


class NetworkDataset(torch.utils.data.Dataset):
    def __init__(self, net_hashes, network_data: BaseNetworkData, transform=None):
        self.net_hashes = net_hashes
        self.network_data = network_data
        self.transform = transform

    def load(self):
        if isinstance(self.net_hashes, str):
            data = pd.read_csv(self.net_hashes)
            self.net_hashes = data['hashes']

        self.network_data.load()

    def __len__(self):
        return len(self.net_hashes)

    def __getitem__(self, index):
        data = self.network_data.get_data(self.net_hashes[index])
        if self.transform is not None:
            data = self.transform(data)

        return data


def _load_data_or_iterable(data, init_func):
    if isinstance(data, list):
        return [init_func(v) for v in data]
    if isinstance(data, dict) and 'hashes' not in data:
        return {k: init_func(v) for k, v in data.items()}
    return init_func(data)


def _not_dict_and_list(data):
    return not isinstance(data, dict) and not isinstance(data, list)

def _to_dict_or_list(data, type):
    assert type == dict or type == list
    if isinstance(data, type):
        return data

    if _not_dict_and_list(data):
        data = [data]

    if type == dict:
        return {i: d for i, d in enumerate(data)}
    return data


def join_dataset_iterables(d1, d2):
    if d1 is None:
        return d2

    if d2 is None:
        return d1

    if _not_dict_and_list(d1) and _not_dict_and_list(d2):
        return [d1, d2]

    if isinstance(d1, dict) or isinstance(d2, dict):
        d1 = _to_dict_or_list(d1, dict)
        d2 = _to_dict_or_list(d2, dict)
        if len(set(d1.keys()).intersection(set(d2.keys()))) > 0:
            d1 = {f"{k}_0": v for k, v in d1.items()}
            d2 = {f"{k}_1": v for k, v in d2.items()}
        return {**d1, **d2}

    if isinstance(d1, list) or isinstance(d2, list):
        d1 = _to_dict_or_list(d1, list)
        d2 = _to_dict_or_list(d2, list)
        return d1 + d2


def join_path(p, base_dir=None):
    return p if base_dir is None else os.path.join(base_dir, p)


def load_from_cfg(data_cfg, init_func):
    keys = ['train', 'val', 'test']
    res = {k: _load_data_or_iterable(data_cfg[k], init_func)
           for k in keys if k in data_cfg}

    return res
