import click
import os
import pandas as pd
import torch

from info_nas.datasets.io_dataset import create_io_data
from info_nas.datasets.search_spaces import NasbenchIOExtractor

from nasbench_pytorch.datasets.cifar10 import prepare_dataset

from searchspace_train.datasets.nasbench101 import PretrainedNB101
from searchspace_train.base import enumerate_trained_networks
from searchspace_train.datasets.nasbench101 import load_nasbench


def load_pretrained_nb101(nb, dataset_dir, dataset_name):
    saved_dataset = pd.read_csv(os.path.join(dataset_dir, dataset_name), index_col=0)
    return PretrainedNB101(nb, net_data=saved_dataset, as_basename=True, training=False)


def load_cifar10(key, batch_size, num_workers, root, random_state):
    workers = {}
    if key == 'train':
        workers['num_workers'] = num_workers
    elif key == 'validation':
        workers['num_val_workers'] = num_workers
    elif key == 'test':
        workers['num_test_workers'] = num_workers
        workers['test_batch_size'] = batch_size
    else:
        raise ValueError(f"Invalid dataset key: {key}, possible: train, validation, test")

    return prepare_dataset(batch_size, random_state=random_state, set_global_seed=True,
                           root=root, **workers)


@click.command()
@click.option('save_path')
@click.option('dataset_dir')
@click.option('--nasbench', default='../../data/nasbench_only108.tfrecord')
@click.option('--dataset_name', default='dataset.csv')
@click.option('--root', default='../../data/cifar/')
@click.option('--batch_size', default=32)
@click.option('--random_state', default=1)
@click.option('--num_workers', default=0)
@click.option('--device', default='cpu')
@click.option('--key', default='validation')
def main(save_path, dataset_dir, nasbench, dataset_name, root, batch_size, random_state, num_workers, device, key):
    nb = load_nasbench(nasbench)
    pnb = load_pretrained_nb101(nb, dataset_dir, dataset_name)
    cifar = load_cifar10(key, batch_size, num_workers, root, random_state)

    # extract IO data
    nb_ex = NasbenchIOExtractor()
    io_data = create_io_data(enumerate_trained_networks(pnb, dir_path=dataset_dir, device=device), nb_ex, cifar[key])

    torch.save(io_data, save_path)


if __name__ == "__main__":
    main()
