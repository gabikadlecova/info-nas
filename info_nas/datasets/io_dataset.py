import torch
import torch.utils.data

from info_nas.datasets.base import BaseIOExtractor, NetworkDataset, NetworkDataModule, BaseNetworkData
from typing import Dict


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
        if self.data is None:
            self.data = torch.load(self.load_path)

    def get_data(self, net_hash):
        return self.data['networks'][net_hash]

    def get_inputs(self, inputs):
        return self.data['images'][inputs]


class IODataset(NetworkDataset):
    def __init__(self, net_hashes, network_data, io_data, transform=None, label_transform=None):
        super().__init__(net_hashes, network_data, transform=transform)
        self.io_data = io_data
        self.label_transform = label_transform
        self.map_idx = None

    def load(self):
        super().load()
        self.io_data.load()

        if self.map_idx is not None:
            return

        # create a map from dataset idx to network output idx
        self.map_idx = []
        for net_hash in self.net_hashes:
            vals = self.io_data.get_data(net_hash)
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
