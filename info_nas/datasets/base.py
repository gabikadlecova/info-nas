from abc import abstractmethod

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split


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


class NetworkDataModule(pl.LightningDataModule):
    def __init__(self, network_data: BaseNetworkData, transform=None, batch_size=32, dataset_class=None,
                 train_hash_list=None, val_hash_list=None, test_hash_list=None, use_val=True, use_test=False, seed=None,
                 val_size=0.1, test_size=0.1):
        super().__init__()
        self.dataset_class = dataset_class if dataset_class is not None else NetworkDataset

        self.network_data = network_data
        self.transform = transform
        self.batch_size = batch_size

        self.seed = seed

        assert train_hash_list is not None or val_hash_list is not None or test_hash_list is not None
        self.train_hash_list = train_hash_list
        self.val_hash_list, self.test_hash_list = None, None
        self.train_set, self.val_set, self.test_set = None, None, None

        if use_val:
            self.val_hash_list = self._split_off_hashes(hashes=val_hash_list, size=val_size, seed=seed)
        if use_test:
            self.test_hash_list = self._split_off_hashes(hashes=test_hash_list, size=test_size, seed=seed)

    def prepare_data(self):
        self.network_data.load()

    def setup(self, stage: str):
        if self.train_hash_list is not None:
            self.train_set = self.dataset_class(self.train_hash_list, self.network_data, transform=self.transform)

        if self.val_hash_list is not None:
            self.val_set = self.dataset_class(self.val_hash_list, self.network_data, transform=self.transform)

        if self.test_hash_list is not None:
            self.test_set = self.dataset_class(self.test_hash_list, self.network_data, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return [DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)]

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

    def _split_off_hashes(self, hashes=None, size=None, seed=None):
        if self.train_hash_list is None:
            return hashes

        # predetermined split
        if hashes is not None:
            hash_set = set(hashes)
            self.train_hash_list = [h for h in self.train_hash_list if h not in hash_set]
            return hashes

        # random split
        train_size = (1.0 - size) if size < 1.0 else (len(self.train_hash_list) - size)
        if seed is not None:
            seed = torch.Generator().manual_seed(seed) if seed is not None else None
            self.train_hash_list, res = random_split(self.train_hash_list, [train_size, size], generator=seed)
        else:
            self.train_hash_list, res = random_split(self.train_hash_list, [train_size, size])


class NetworkDataset(torch.utils.data.Dataset):
    def __init__(self, hash_list, network_data: BaseNetworkData, transform=None):
        self.hash_list = hash_list
        self.network_data = network_data
        self.transform = transform

    def __len__(self):
        return len(self.hash_list)

    def __getitem__(self, index):
        data = self.network_data.get_data(self.hash_list[index])
        if self.transform is not None:
            data = self.transform(data)

        return data
