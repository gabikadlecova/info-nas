from info_nas.datasets.base import NetworkDataset


class AccuracyDataset(NetworkDataset):
    def __init__(self, network_data, labeled_data, transform=None, label_transform=None):
        super().__init__(network_data, transform=transform)
        self.labeled_data = labeled_data
        self.label_transform = label_transform

        self.data_len = len(labeled_data)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        net_data = super().__getitem__(index)

        acc = self.labeled_data[self.hash_list[index]]
        net_data['outputs'] = acc

        if self.label_transform is not None:
            net_data = self.transform(net_data)

        return net_data
