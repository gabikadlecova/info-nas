import pickle

import click
import numpy as np
import torch
import torchvision
from nasbench import api

from info_nas.config import local_dataset_cfg
from info_nas.datasets.io.transforms import IncludeBias, load_scaler, SortByWeights, after_scale_path
from info_nas.datasets.io.semi_dataset import labeled_network_dataset
from info_nas.datasets.arch2vec_dataset import prepare_labeled_dataset


@click.command()
@click.argument('scale_name')
@click.argument('scale_path')
@click.option('--dataset', default='../data/train_labeled.pt')
@click.option('--nasbench_path', default='../data/nasbench.pickle')
@click.option('--axis', default=None, type=int)
@click.option('--axis_bef', default=None, type=int)
@click.option('--normalize_bef/--minmax_bef', default=True)
@click.option('--include_bias/--no_bias', default=True)
@click.option('--config', default=None)
def main(scale_name, scale_path, dataset, nasbench_path, axis, axis_bef, normalize_bef, include_bias, config):
    if nasbench_path.endswith('.pickle'):
        with open(nasbench_path, 'rb') as f:
            nb = pickle.load(f)
    else:
        nb = api.NASBench(nasbench_path)

    if config is None:
        config = local_dataset_cfg

    transforms = []

    if include_bias:
        assert 'include_bias' in scale_path
        transforms.append(IncludeBias())

    scaler = load_scaler(scale_path, normalize_bef, axis_bef, include_bias)
    transforms.append(scaler)

    transforms.append(SortByWeights())
    transforms = torchvision.transforms.Compose(transforms)

    key = 'val' if scale_name == 'valid' else scale_name
    dataset, _ = prepare_labeled_dataset(dataset, nb, key=key, remove_labeled=False, config=config)
    dataset = labeled_network_dataset(dataset, transforms=transforms)

    print("Loading output dataset...")
    out_data = np.array([item['output'].numpy() for item in dataset])
    mean = np.mean(out_data, axis=axis)
    std = np.std(out_data, axis=axis)

    out_path = after_scale_path(scale_path, axis)
    data = {
        'mean': mean,
        'std': std
    }

    with open(out_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
