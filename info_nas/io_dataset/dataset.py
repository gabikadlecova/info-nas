import torch.utils.data
from searchspace_train.base import enumerate_trained_networks, BaseDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from info_nas.io_dataset.base import BaseIOExtractor


class SemiIODataset:
    def __init__(self, name, io_data, network_data, k=1, batch_size=32, shuffle=True, io_tranforms=None,
                 unlabeled_transforms=None, **kwargs):
        self.name = name
        self.k = k

        io_net_data, unlabeled_net_data = split_by_hash_set(network_data, io_data)

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
            yield data, True

    def _next_unlabeled(self):
        for _ in range(self.k):
            for data in self.unlabeled_loader:
                yield data, False


def create_io_data(search_space: BaseDataset, extractor: BaseIOExtractor, dataset, save_input_images: bool = True,
                   device: str = None, **kwargs):
    result = {}

    if save_input_images:
        images = [b[0] for b in dataset]
        images = torch.cat(images)
        result['images'] = images

    for net_hash, net in enumerate_trained_networks(search_space, device=device, **kwargs):
        io_data = extractor.get_io_data(net, dataset)

        result[net_hash] = io_data

    return result


def split_by_hash_set(data_dict, hash_set):
    contains, rest = {}, {}
    for k, v in data_dict.items():
        if k in hash_set:
            contains[k] = v
        else:
            rest[k] = v

    return contains, rest


def net_train_test_split(network_data, io_data=None, random_state=None, test_size=None, **kwargs):
    hash_keys = list(network_data.keys())
    train_hashes, _ = train_test_split(hash_keys, test_size=test_size, random_state=random_state, **kwargs)

    return net_hash_set_split(network_data, train_hashes, io_data=io_data)


def net_hash_set_split(network_data, hash_set, io_data=None):
    train_data, test_data = split_by_hash_set(network_data, hash_set)
    if io_data is None:
        return train_data, test_data

    train_io, test_io = split_by_hash_set(io_data, hash_set)

    return (train_data, train_io), (test_data, test_io)


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


class IODataset(NetworkDataset):
    def __init__(self, network_data, io_data, transform=None, io_transform=None):
        super().__init__(network_data, transform=transform)
        self.io_data = io_data
        self.io_transform = io_transform

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
        if self.io_transform is not None:
            net_data = self.transform(net_data)

        return net_data
