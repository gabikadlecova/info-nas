import torch.utils.data

from info_nas.datasets.base import BaseIOExtractor, NetworkDataset
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


class IODataset(NetworkDataset):
    def __init__(self, hash_list, network_data, io_data, transform=None, label_transform=None):
        super().__init__(hash_list, network_data, transform=transform)
        self.io_data = io_data
        self.label_transform = label_transform

        # create a map from dataset idx to network output idx
        self.map_idx = []
        for k, vals in io_data['networks'].items():
            for i, _ in enumerate(vals['outputs']):
                self.map_idx.append((k, i))

    def __len__(self):
        return len(self.map_idx)

    def __getitem__(self, index):
        net_hash, image_index = self.map_idx[index]

        # get io data
        net_io = self.io_data['networks'][net_hash]
        data = {'weights': net_io['weights'], 'biases': net_io['biases']}
        for k in ['inputs', 'outputs', 'labels']:
            data[k] = net_io[k][image_index]

        # replace index with input image
        data['inputs'] = self.io_data['images'][data['inputs']]  # TODO test this

        # get adj and ops
        data.update(self.network_data.get_data(net_hash))

        if self.label_transform is not None:
            data = self.label_transform(data)

        return data
