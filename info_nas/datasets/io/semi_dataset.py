import math
import warnings

import torch
import torch.utils.data


def get_train_valid_datasets(labeled, unlabeled, k=1, repeat_unlabeled=1, batch_size=32, n_workers=0, shuffle=True,
                             val_batch_size=100, n_valid_workers=0, labeled_transforms=None,
                             labeled_val_transforms=None, **kwargs):

    train_labeled = labeled_network_dataset(labeled['train'], transforms=labeled_transforms)
    valid_labeled = labeled_network_dataset(labeled['valid'], transforms=labeled_val_transforms)

    train_unlabeled = unlabeled_network_dataset(unlabeled['train'])
    valid_unlabeled = unlabeled_network_dataset(unlabeled['val'])

    n_labeled = len(labeled['train']['net_repo'])
    train_dataset = SemiSupervisedDataset(train_labeled, train_unlabeled, n_labeled, k=k, batch_size=batch_size,
                                          repeat_unlabeled=repeat_unlabeled, n_workers=n_workers,
                                          shuffle=shuffle, **kwargs)

    valid_labeled_dataset = torch.utils.data.DataLoader(valid_labeled, batch_size=val_batch_size,
                                                        num_workers=math.floor(n_valid_workers / 2), **kwargs)
    valid_unlabeled_dataset = torch.utils.data.DataLoader(valid_unlabeled, batch_size=val_batch_size,
                                                          num_workers=math.ceil(n_valid_workers / 2), **kwargs)

    return train_dataset, valid_labeled_dataset, valid_unlabeled_dataset


def labeled_network_dataset(labeled, transforms=None, return_hash=True, return_ref_id=False):
    net_repo = labeled['net_repo']

    # indexing in the original input (io dataset uses input id 0)
    ref_dataset = (labeled['dataset'], labeled['labels']) if labeled['use_reference'] else None

    return ReferenceNetworkDataset(labeled['net_hashes'], labeled['inputs'], labeled['outputs'],
                                   reference_dataset=ref_dataset, net_repo=net_repo,
                                   transform=transforms, return_hash=return_hash, return_ref_id=return_ref_id)


def unlabeled_network_dataset(dataset):
    _, adj, ops, _ = dataset
    return NetworkDataset(adj, ops)


# TODO if larger dataset, load from file (IterableDataset - only for io, unlabeled are short enough)
class NetworkDataset(torch.utils.data.Dataset):
    def __init__(self, *args):
        super().__init__()

        self.data = args

        self.data_len = len(args[0])
        assert all(len(a) == self.data_len for a in args)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        item = [a[index] for a in self.data]
        return item


class ReferenceNetworkDataset(NetworkDataset):
    def __init__(self, *args, reference_dataset=None, reference_id=1, net_repo=None, net_id=0,
                 return_hash=True, return_ref_id=False, transform=None):

        super().__init__(*args)

        self.reference_dataset = reference_dataset[0] if reference_dataset is not None else None
        self.reference_labels = reference_dataset[1] if reference_dataset is not None else None
        self.reference_id = reference_id

        self.net_repo = net_repo
        self.net_id = net_id

        self.return_hash = return_hash
        self.return_ref_id = return_ref_id

        self._batch_names = self.get_batch_names()
        self.transform = transform

    def get_batch_names(self):
        """
        Returns a list of names that describe the batch (the dataset can optionally contain labels, weights and
        biases).

        Returns:
            Names of each member of the batch tuple.

        """
        names = ['adj', 'ops', 'input', 'output']

        if self.reference_dataset is not None:
            names.append('label')

        if self.net_repo is not None:
            names.append('weights')
            names.append('bias')

        if self.return_hash:
            names.append('hash')

        if self.return_ref_id:
            names.append('ref_id')

        return names

    def __getitem__(self, index):
        item = super().__getitem__(index)

        # get inputs from the reference dataset
        if self.reference_dataset is not None:
            ref_id = item[self.reference_id]

            data = self.reference_dataset[ref_id]
            label = self.reference_labels[ref_id]
            item[self.reference_id] = data
            item.append(label)

        # get network metadata
        if self.net_repo is not None:
            net_hash = item[self.net_id]
            net_entry = self.net_repo[net_hash]

            # replace hash entry with adj, ops
            item[self.net_id] = net_entry['adj']
            item.insert(self.net_id + 1, net_entry['ops'])

            # additional info goes to the end
            item.append(net_entry['weights'])
            item.append(net_entry['bias'])

            if self.return_hash:
                item.append(net_hash)

        if self.return_ref_id and self.reference_dataset is not None:
            item.append(ref_id)

        item = {name: v for name, v in zip(self._batch_names, item)}

        if self.transform is not None:
            item = self.transform(item)

        return item


class SemiSupervisedDataset:
    def __init__(self, labeled, unlabeled, n_labeled_nets, k=1, batch_size=32, n_workers=0, shuffle=True,
                 repeat_unlabeled=1, **kwargs):
        self.n, self.n_labeled, self.n_unlabeled = 0, 0, 0

        # datasets and their iterators
        self.labeled = torch.utils.data.DataLoader(labeled, batch_size=batch_size, shuffle=shuffle,
                                                   num_workers=math.floor(n_workers / 2), **kwargs)
        self.unlabeled = torch.utils.data.DataLoader(unlabeled, batch_size=batch_size, shuffle=shuffle,
                                                     num_workers=math.ceil(n_workers / 2), **kwargs)
        self.labeled_iter = None
        self.unlabeled_iter = None

        # max values for iteration
        self.max_labeled = len(self.labeled)
        self.max_unlabeled = len(self.unlabeled) * repeat_unlabeled
        self.max_n = self.max_labeled + self.max_unlabeled

        # values that determine when to draw unlabeled vs labeled batches
        n_unlabeled_nets = len(self.unlabeled)
        n_labeled_orig = n_labeled_nets
        n_labeled_nets = n_labeled_nets // batch_size

        if n_labeled_nets == 0:
            warnings.warn(f"The number of labeled nets is less than the batch size ({n_labeled_orig} vs {batch_size}).")
            n_labeled_nets = 1

        self.k = k

        self.labeled_coef = n_unlabeled_nets // n_labeled_nets
        assert self.labeled_coef >= 1, f"There cannot be more labeled nets than unlabeled " \
                                       f"({n_labeled_nets} vs {n_unlabeled_nets})."

    def __len__(self):
        return self.max_n

    def __iter__(self):
        self.n, self.n_labeled, self.n_unlabeled = 0, 0, 0
        self.labeled_iter = iter(self.labeled)
        self.unlabeled_iter = iter(self.unlabeled)
        return self

    def __next__(self):
        if self.n < self.max_n:
            self.n += 1

            # return next item, alternate labeled and unlabeled
            if self.should_choose_labeled():
                self.n_labeled += 1
                return next(self.labeled_iter)
            else:
                self.n_unlabeled += 1
                return self.next_unlabeled()
        else:
            raise StopIteration()

    def next_unlabeled(self):
        """
        Return the next batch of the unlabeled dataset, start the iteration over if the end is reached.

        Returns:
            Next batch from the unlabeled dataset.
        """
        try:
            return next(self.unlabeled_iter)
        except StopIteration:
            self.unlabeled_iter = iter(self.unlabeled)
            return next(self.unlabeled_iter)

    def should_choose_labeled(self):
        """
        Determines how to draw the next batch (labeled vs unlabeled)

        - coef is the proportion of unlabeled network to labeled network batches
            - there are more labeled batches than labeled network batches (one network occurs multiple times in the
              labeled dataset)
        - curr_u = (n_unlabeled / coef) + 1
        - if (self.k * curr_u) > n_labeled, return labeled batches, otherwise unlabeled

        If one of the iterators is exhausted, the batches are returned from the other one till exhaustion.

        Example:
          - if there are 3 labeled networks and 48 unlabeled, coef is 48 // 3 == 16
          - if coef is 16 and k is 1, 1 labeled batch is followed by 16 unlabeled
          - if coef is 16 and k is 5, 5 labeled batches are followed by 16 unlabeled

        Returns: True if the next batch should be drawn from the labeled dataset, False for unlabeled.

        """
        if self.n_labeled >= self.max_labeled:
            return False

        if self.n_unlabeled >= self.max_unlabeled:
            return True

        curr_u = self.n_unlabeled // self.labeled_coef + 1
        return (self.k * curr_u) > self.n_labeled
