from abc import abstractmethod

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class BaseIOExtractor:
    @abstractmethod
    def get_io_data(self, net, data):
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


def split_by_hash_set(data_dict, hash_set):
    contains, rest = {}, {}
    for k, v in data_dict.items():
        if k in hash_set:
            contains[k] = v
        else:
            rest[k] = v

    return contains, rest


def net_train_test_split(network_data, labeled_data=None, random_state=None, test_size=None, **kwargs):
    hash_keys = list(network_data.keys())
    train_hashes, _ = train_test_split(hash_keys, test_size=test_size, random_state=random_state, **kwargs)

    return net_hash_set_split(network_data, train_hashes, io_data=labeled_data)


def net_hash_set_split(network_data, hash_set, io_data=None):
    train_data, test_data = split_by_hash_set(network_data, hash_set)
    if io_data is None:
        return train_data, test_data

    train_io, test_io = split_by_hash_set(io_data, hash_set)

    return (train_data, train_io), (test_data, test_io)


class SemiSupervisedDataset:
    def __init__(self, name, labeled_class, labeled_data, network_data, k=1, batch_size=32, shuffle=True,
                 label_transform=None, transform=None, **kwargs):
        self.name = name
        self.k = k

        labeled_net_data, unlabeled_net_data = split_by_hash_set(network_data, labeled_data)

        self.io_dataset = labeled_class(labeled_net_data, labeled_data, transform=transform,
                                        label_transform=label_transform)
        self.unlabeled_dataset = NetworkDataset(unlabeled_net_data, transform=transform)

        self.io_loader = DataLoader(self.io_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.unlabeled_loader = DataLoader(self.unlabeled_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

        self.n_labeled = len(self.io_loader) // (k * len(self.unlabeled_loader))
        assert self.n_labeled > 0

    def __iter__(self):
        labeled_iter = self._next_labeled()
        unlabeled_iter = self._next_unlabeled()

        while True:
            for _ in range(self.n_labeled):
                yield self._next_batch(labeled_iter, unlabeled_iter)
            yield self._next_batch(unlabeled_iter, labeled_iter)

    @staticmethod
    def _next_batch(first_it, backup_it):
        try:
            yield next(first_it)
        except StopIteration:
            while True:
                yield next(backup_it)

    def _next_labeled(self):
        for data in self.io_loader:
            yield data, True

    def _next_unlabeled(self):
        for _ in range(self.k):
            for data in self.unlabeled_loader:
                yield data, False


class NetworkDataset(torch.utils.data.Dataset):
    def __init__(self, network_data, transform=None):
        self.network_data = network_data
        self.hash_list = list(network_data.keys())
        self.transform = transform

    def __len__(self):
        return len(self.network_data)

    def __getitem__(self, index):
        data = self.network_data[self.hash_list[index]]
        if self.transform is not None:
            data = self.transform(data)

        return data
