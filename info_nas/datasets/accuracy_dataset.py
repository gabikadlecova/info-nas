from info_nas.datasets.base import NetworkDataset


class AccuracyDataset(NetworkDataset):
    def __init__(self, hash_list, network_data, io_data, transform=None, label_transform=None,
                 acc_key='final_validation_accuracy'):
        super().__init__(hash_list, network_data, transform=transform)
        self.label_transform = label_transform
        self.acc_key = acc_key
        self.io_data = io_data

    def __len__(self):
        return len(self.hash_list)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        io_data = self.io_data.get_data(self.hash_list[index])

        net_data = {'adj': data['adj'], 'ops': data['ops'], 'outputs': io_data[self.acc_key]}

        if self.label_transform is not None:
            net_data = self.transform(net_data)

        return net_data
