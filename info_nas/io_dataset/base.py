import torch

from abc import abstractmethod
from searchspace_train.base import BaseDataset, enumerate_trained_networks


# TODO spíš jako torch.dataset
class IODataset:
    def __init__(self, name, io_data, network_data):
        self.name = name
        self.io_data = io_data
        self.network_data = network_data
        # preprocessed - maybe initially raw nb data, then prepro. Or create on demand.
        # so we'll have a function "load net data", then apply arch2vec prepro...


class BaseIOExtractor:
    @abstractmethod
    def get_io_data(self, net, data):
        pass


def create_io_dataset(search_space: BaseDataset, extractor: BaseIOExtractor, dataset, save_input_images: bool = True,
                      device: str = None, **kwargs):
    result = {}

    if save_input_images:
        images = [b for b in dataset]
        images = torch.cat(images)
        result['images'] = images

    for net_hash, net in enumerate_trained_networks(search_space, device=device, **kwargs):
        io_data = extractor.get_io_data(net, dataset)

        result[net_hash] = io_data

    return result


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
