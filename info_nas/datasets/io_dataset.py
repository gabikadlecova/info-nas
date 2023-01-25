import pickle

import torch.utils.data

from info_nas.datasets.base import BaseIOExtractor, NetworkDataset, NetworkDataModule, BaseNetworkData
from typing import Dict, List


def create_io_data(networks: Dict[str, torch.nn.Module], extractor: BaseIOExtractor, dataset,
                   save_input_images: bool = True):
    result = {'networks': {}}

    if save_input_images:
        images = [b[0] for b in dataset]
        images = torch.cat(images)
        result['images'] = images

    for net_hash, net in networks:
        io_data = extractor.get_io_data(net, dataset)

        result['networks'][net_hash] = io_data

    return result


class IOData:
    def __init__(self, data=None, load_path=None):
        assert data is not None or load_path is not None
        self.data = data
        self.load_path = load_path

    def load(self):
        with open(self.load_path, 'rb') as f:
            self.data = pickle.load(self.load_path)

    def get_data(self, net_hash):
        return self.data['networks'][net_hash]

    def get_inputs(self, inputs):
        return self.data['images'][inputs]


class IODataModule(NetworkDataModule):
    def __init__(self, network_data: BaseNetworkData, io_data, dataset_class=None,
                 label_transform=None, **kwargs):
        self.dataset_class = dataset_class if dataset_class is not None else IODataset
        super().__init__(network_data, dataset_class=dataset_class, **kwargs)

        self.label_transform = label_transform
        self.io_data = io_data

    def prepare_data(self):
        super().prepare_data()
        if self.io_data is not self.network_data:
            self.io_data.load()

    def setup(self, stage: str):
        if self.train_hash_list is not None:
            self.train_set = self.dataset_class(self.train_hash_list, self.network_data, self.io_data,
                                                transform=self.transform, label_transform=self.label_transform)

        if self.val_hash_list is not None:
            self.val_set = self.dataset_class(self.val_hash_list, self.network_data, self.io_data,
                                              transform=self.transform, label_transform=self.label_transform)
        if self.test_hash_list is not None:
            self.test_set = self.dataset_class(self.test_hash_list, self.network_data, self.io_data,
                                               transform=self.transform, label_transform=self.label_transform)


class IODataset(NetworkDataset):
    def __init__(self, hash_list, network_data, io_data, transform=None, label_transform=None):
        super().__init__(hash_list, network_data, transform=transform)
        self.io_data = io_data
        self.label_transform = label_transform

        # create a map from dataset idx to network output idx
        self.map_idx = []
        for net_hash in hash_list:
            vals = io_data['networks'][net_hash]
            for i, _ in enumerate(vals['outputs']):
                self.map_idx.append((net_hash, i))

    def __len__(self):
        return len(self.map_idx)

    def __getitem__(self, index):
        net_hash, image_index = self.map_idx[index]

        # get io data
        net_io = self.io_data.get_data(net_hash)
        data = {'weights': net_io['weights'], 'biases': net_io['biases']}
        for k in ['inputs', 'outputs', 'labels']:
            data[k] = net_io[k][image_index]

        # replace index with input image
        data['inputs'] = self.io_data.get_inputs(data['inputs'])  # TODO test this

        # get adj and ops
        data.update(self.network_data.get_data(net_hash))

        if self.label_transform is not None:
            data = self.label_transform(data)

        return data
