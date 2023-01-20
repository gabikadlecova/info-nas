from abc import abstractmethod
from typing import List

import torch
import pytorch_lightning as pl
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


class NetworkDataModule(pl.LightningDataModule):
    def __init__(self, hash_list: List[str], network_data: BaseNetworkData, transform=None, val_hash_list=None,
                 test_hash_list=None, use_test=False):
        super().__init__()
        self.network_data = network_data
        self.transform = transform

        self.hash_list = hash_list
        self.val_hash_list = self._split_off_hashes(val_hash_list) if val_hash_list is not None else None
        self.test_hash_list = self._split_off_hashes(test_hash_list) if test_hash_list is not None else None

    def prepare_data(self):
        self.network_data.load()

    def setup(self, stage: str):
        #TODO split
        pass

    def _split_off_hashes(self, hashes):
        hash_set = set(hashes)
        self.hash_list = [h for h in self.hash_list if h not in hash_set]
        return hashes

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass


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
