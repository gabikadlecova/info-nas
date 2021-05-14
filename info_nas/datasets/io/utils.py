import click

import os
import numpy as np
import pickle


def _load_dataset(data_path):

    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    return dataset['data'], np.array(dataset['labels'])


def _choose_n_labels(data, labels, n=120):
    index = labels[labels <= n]

    return data[index], labels[index]


def create_ImageNet_16_n(train_path, val_path, save_dir, n):
    train_data, train_labels = _load_dataset(train_path)  #TODO ten train je rozchunkovanej
    val_data, val_labels = _load_dataset(val_path)

    train_data, train_labels = _choose_n_labels(train_data, train_labels, n=n)
    val_data, val_labels = _choose_n_labels(val_data, val_labels, n=n)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    np.savez(os.path.join(save_dir, f'train_{n}.npz'), train_data, train_labels)
    np.savez(os.path.join(save_dir, f'val_{n}.npz'), val_data, val_labels)


@click.command()
@click.option('train_path')
@click.option('val_path')
@click.argument('--n', default=120)
@click.argument('--save_dir', default='../data/ImageNet-16/')
def load_data(train_path, val_path, n, save_dir):
    create_ImageNet_16_n(train_path, val_path, save_dir, n=n)


if __name__ == "__main__":
    load_data()
