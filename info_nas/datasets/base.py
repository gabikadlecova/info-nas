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


def datasets_to_iterable(data):
    if isinstance(data, dict) or isinstance(data, list):
        return data
    return [data]


def apply_to_vals(dataset, func):
    if isinstance(dataset, dict):
        return {k: func(v) for k, v in dataset.items()}
    return [func(v) for v in dataset]


class NetworkDataModule(pl.LightningDataModule):
    def __init__(self, network_data: BaseNetworkData, train_datasets, val_datasets=None, test_datasets=None,
                 batch_size=32):
        super().__init__()

        self.network_data = network_data
        self.batch_size = batch_size

        self.data = {'train': datasets_to_iterable(train_datasets)}
        if val_datasets is not None:
            self.data['val'] = datasets_to_iterable(val_datasets)
        if test_datasets is not None:
            self.data['test'] = datasets_to_iterable(test_datasets)

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
        shuffle = True if key == 'train' else False
        return apply_to_vals(self.data[key], lambda d: DataLoader(d, batch_size=self.batch_size, shuffle=shuffle))


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
