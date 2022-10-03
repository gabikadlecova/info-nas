import torch
import torch.utils.data

from abc import abstractmethod
from searchspace_train.base import BaseDataset, enumerate_trained_networks
from torch.utils.data import DataLoader


# TODO chci nějak funkci dole předělat, aby ještě brala nelabelovaný arch atd,
#    ale nevím, jestli to dát do searchspace_train nebo nějak mimo. Problém by byly search spacy bez
#    unlabeled dat.


class SemiIODataset:
    def __init__(self, name, io_data, network_data, k=1, batch_size=32, shuffle=True, **kwargs):
        self.name = name
        self.k = k

        io_net_data = {k: d for k, d in network_data.items() if k in io_data}
        unlabeled_net_data = {k: d for k, d in network_data.items() if k not in io_data}

        self.io_dataset = IODataset(io_net_data, io_data)
        self.unlabeled_dataset = NetworkDataset(unlabeled_net_data)

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
            yield data

    def _next_unlabeled(self):
        for _ in range(self.k):
            for data in self.unlabeled_loader:
                yield data


class NetworkDataset(torch.utils.data.Dataset):
    def __init__(self, network_data):
        self.network_data = network_data
        self.hash_list = list(network_data.keys())

    def __len__(self):
        return len(self.network_data)

    def __getitem__(self, index):
        return self.network_data[self.hash_list[index]]


# preprocessed - maybe initially raw nb data, then prepro. Or create on demand.
# so we'll have a function "load net data", then apply arch2vec prepro...


class IODataset(NetworkDataset):
    def __init__(self, network_data, io_data):
        super().__init__(network_data)
        self.io_data = io_data

        # check if all IO data has the same dimensions
        self.io_len = None
        for v in io_data.values():
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
        net_io = self.io_data[self.hash_list[div_index]]

        image_data = {'weights': net_io['weights'], 'biases': net_io['biases']}
        for k in ['inputs', 'outputs', 'labels']:
            image_data[k] = net_io[k][image_index]

        image_data['inputs'] = self.io_data['images'][image_data['inputs']]  # TODO test this

        net_data.update(image_data)
        return net_data


class BaseIOExtractor:
    @abstractmethod
    def get_io_data(self, net, data):
        pass


def create_io_data(search_space: BaseDataset, extractor: BaseIOExtractor, dataset, save_input_images: bool = True,
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
