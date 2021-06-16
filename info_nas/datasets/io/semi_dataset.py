import torch


# TODO if larger dataset, load from file (IterableDataset - jen pro io, normal jsou v poho)
class NetworkDataset(torch.utils.data.Dataset):
    def __init__(self, *args):
        super().__init__()

        self.data = args

        self.data_len = len(args[0])
        assert all(len(a) == self.data_len for a in args)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        return tuple(a[index] for a in self.data)


class SemiSupervisedDataset:
    def __init__(self, labeled, unlabeled, n_labeled, batch_size=32, n_workers=0, shuffle=False, **kwargs):
        self.n = 0
        # todo max_n

        # TODO střídej dle poměru, co jsem psala (1 : 49 pro 100 labeled 4900 unlabeled, nebo / K protože cifar batche)
        pass

    def __iter__(self):
        pass

    def __next__(self):
        if 0 == self.n:
            pass
        else:
            raise StopIteration()
