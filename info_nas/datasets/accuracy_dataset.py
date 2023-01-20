from info_nas.datasets.base import NetworkDataset


class AccuracyDataset(NetworkDataset):
    def __init__(self, hash_list, network_data, transform=None, label_transform=None):
        super().__init__(hash_list, network_data, transform=transform)
        self.label_transform = label_transform

    def __len__(self):
        return len(self.hash_list)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        net_data = {'adj': data['adj'], 'ops': data['ops'], 'outputs': data['val_accuracy']}

        if self.label_transform is not None:
            net_data = self.transform(net_data)

        return net_data
