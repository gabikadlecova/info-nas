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
    def __init__(self, network_data, io_data, transform=None, label_transform=None):
        super().__init__(network_data, transform=transform)
        self.io_data = io_data
        self.label_transform = label_transform

        # check if all IO data has the same dimensions
        self.io_len = None
        for k, v in io_data['networks'].items():
            if self.io_len is None:
                self.io_len = len(v['outputs'])

            assert len(v['outputs']) == self.io_len

        self.data_len = self.io_len * len(io_data)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        div_index = index // self.io_len
        net_data = super().__getitem__(div_index)

        image_index = index % self.io_len
        net_io = self.io_data['networks'][self.hash_list[div_index]]

        image_data = {'weights': net_io['weights'], 'biases': net_io['biases']}
        for k in ['inputs', 'outputs', 'labels']:
            image_data[k] = net_io[k][image_index]

        image_data['inputs'] = self.io_data['images'][image_data['inputs']]  # TODO test this

        net_data.update(image_data)
        if self.label_transform is not None:
            net_data = self.label_transform(net_data)

        return net_data
