import pickle

import click
import numpy as np
import torch

from info_nas.datasets.arch2vec_dataset import prepare_labeled_dataset
from info_nas.datasets.io.semi_dataset import labeled_network_dataset
from info_nas.datasets.io.transforms import get_transforms
from info_nas.models.losses import losses_dict
from nasbench import api

from info_nas.config import local_dataset_cfg


@click.command()
@click.argument('scale_name')
@click.argument('scale_path')
@click.option('--dataset', default='../data/train_labeled.pt')
@click.option('--nasbench_path', default='../data/nasbench.pickle')
@click.option('--axis', default=None, type=int)
@click.option('--axis_after', default=None, type=int)
@click.option('--batch_size', default=32, type=int)
@click.option('--normalize/--minmax', default=True)
@click.option('--include_bias/--no_bias', default=True)
@click.option('--scale_whole/--no_scale_whole', default=False)
@click.option('--config', default=None)
def main(scale_name, scale_path, dataset, nasbench_path, axis, axis_after, batch_size, normalize, include_bias,
         scale_whole, config):

    if nasbench_path.endswith('.pickle'):
        with open(nasbench_path, 'rb') as f:
            nb = pickle.load(f)
    else:
        nb = api.NASBench(nasbench_path)

    if config is None:
        config = local_dataset_cfg

    transforms = get_transforms(scale_path, include_bias, axis, normalize,
                                scale_whole=scale_whole, axis_whole=axis_after)

    key = 'val' if scale_name == 'valid' else scale_name
    dataset, _ = prepare_labeled_dataset(dataset, nb, key=key, remove_labeled=False, config=config)
    dataset = labeled_network_dataset(dataset, transforms=transforms)

    print("Loading output dataset...")
    out_data = []
    for i, item in enumerate(dataset):
        if i % 10000 == 0:
            print(i)
        out_data.append(item[3].numpy())

    out_data = np.array(out_data)
    print(out_data.shape)
    print(np.min(out_data))
    print(np.max(out_data))
    print(np.mean(out_data))

    loss_stats = {k: [] for k in losses_dict.keys() if k != 'weighted'}

    mean_stats = out_data.mean(axis=0)
    print(mean_stats.shape)
    mean_stats = np.tile(mean_stats, (batch_size, 1))
    mean_stats = torch.Tensor(mean_stats)

    losses = {k: v() for k, v in losses_dict.items() if k != 'weighted'}

    for i in range(len(out_data) // batch_size):
        data = out_data[i:i + batch_size]
        data = torch.Tensor(data)

        if len(data) < batch_size:
            mean_stats = mean_stats[:len(data)]

        for loss_name, loss in losses.items():
            val = loss(data, mean_stats).item()
            loss_stats[loss_name].append(val)

    for loss_name, stats in loss_stats.items():
        print(f"{loss_name}: mean - {np.mean(stats)} | std - {np.std(stats)} | min - {np.min(stats)} | "
              f"median - {np.median(stats)} | max - {np.max(stats)} |")


if __name__ == "__main__":
    main()
