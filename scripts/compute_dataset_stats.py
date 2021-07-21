import os.path
import pickle

import click
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from info_nas.datasets.arch2vec_dataset import prepare_labeled_dataset, split_off_valid
from info_nas.datasets.io.semi_dataset import labeled_network_dataset
from info_nas.datasets.io.transforms import get_transforms, get_all_scales
from info_nas.models.losses import losses_dict
from nasbench import api

from info_nas.config import local_dataset_cfg, load_json_cfg


@click.command()
@click.argument('scale_name')
@click.option('--dataset', default='../data/train_labeled.pt')
@click.option('--scale_dir', default='../data/scales/')
@click.option('--scale_cfg', default='../configs/scale_test_config.json')
@click.option('--nasbench_path', default='../data/nasbench.pickle')
@click.option('--batch_size', default=32)
@click.option('--split_ratio', default=None, type=float)
@click.option('--config', default=None)
@click.option('--save_dir', default=None)
@click.option('--use_larger_part/--use_smaller_part', default=False)
def main(scale_name, dataset, scale_dir, scale_cfg, nasbench_path, batch_size, split_ratio, config, save_dir,
         use_larger_part):

    dataset_name = dataset

    if nasbench_path.endswith('.pickle'):
        with open(nasbench_path, 'rb') as f:
            nb = pickle.load(f)
    else:
        nb = api.NASBench(nasbench_path)

    if config is None:
        config = local_dataset_cfg

    scale_cfg = load_json_cfg(scale_cfg)

    # load all scaling
    scale_config = scale_cfg["scale"]
    include_bias = scale_config["include_bias"]
    normalize = scale_config["normalize"]
    multiply_by_weights = scale_config["multiply_by_weights"]
    use_scale_whole = scale_config["scale_whole"]

    for k, v in scale_config.items():
        print(f"{k}: {v}")

    scale_train, scale_valid, scale_whole = get_all_scales(scale_dir, scale_config)
    scale_whole = scale_whole if use_scale_whole else None
    print(f"Scale paths: {scale_train}, {scale_valid}, {scale_whole}")

    transforms = get_transforms(scale_train if scale_name == "train" else scale_valid, include_bias, normalize,
                                multiply_by_weights, scale_whole_path=scale_whole)

    key = 'val' if scale_name == 'valid' else scale_name
    dataset, _ = prepare_labeled_dataset(dataset, nb, key=key, remove_labeled=False, config=config)
    if split_ratio is not None:
        larger_part, dataset = split_off_valid(dataset, ratio=split_ratio)
        dataset = larger_part if use_larger_part else dataset

    dataset = labeled_network_dataset(dataset, transforms=transforms)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=scale_name == "train",
                                              num_workers=0)

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

    for batch in data_loader:
        data = batch[3]

        if len(data) < batch_size:
            mean_stats = mean_stats[:len(data)]

        for loss_name, loss in losses.items():
            val = loss(data, mean_stats).item()
            loss_stats[loss_name].append(val)

    df = []

    for loss_name, stats in loss_stats.items():
        print(f"{loss_name}: mean - {np.mean(stats)} | std - {np.std(stats)} | min - {np.min(stats)} | "
              f"median - {np.median(stats)} | max - {np.max(stats)} |")

        df.append({
            'loss_name': loss_name,
            'mean': np.mean(stats),
            'std': np.std(stats),
            'min': np.min(stats),
            'max': np.max(stats),
            'median': np.median(stats),
        })

    if save_dir is not None:
        df = pd.DataFrame(df)
        out_name = os.path.basename(dataset_name).replace('.pt', '')
        df.to_csv(os.path.join(save_dir, f'{out_name}_baseline.csv'))


if __name__ == "__main__":
    main()
