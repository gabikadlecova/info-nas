import random

import click
import json
import os

import numpy as np
import torch

from info_nas.config import local_dataset_cfg, load_json_cfg
from info_nas.datasets.networks.pretrained import pretrain_network_dataset
from nasbench import api
from nasbench_pytorch.datasets.cifar10 import prepare_dataset
from scripts.utils import mkdir_if_not_exists


@click.command()
@click.argument('hash')
@click.option('--nasbench_path', default='../data/nasbench_only108.tfrecord')
@click.option('--config_path', default='../configs/pretrain_config.json')
@click.option('--out_dir', default='.')
@click.option('--root', default='../data/cifar/')
@click.option('--seed', default=1)
@click.option('--n_seeds', default=10)
@click.option('--device', default='cuda')
def main(hash, nasbench_path, config_path, out_dir, root, seed, n_seeds, device):
    device = torch.device(device)

    out_dir = os.path.join(out_dir, f"out_{hash}/")
    mkdir_if_not_exists(out_dir)

    # pretrain

    if not len(config_path) or config_path is None:
        config = local_dataset_cfg
    else:
        config = load_json_cfg(config_path)

    # save config for reference
    config_name = os.path.basename(config_path) if config_path is not None else 'config.json'
    with open(os.path.join(out_dir, config_name), 'w+') as f:
        json.dump(config, f, indent='    ')

    nasbench = api.NASBench(nasbench_path)
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = prepare_dataset(root=root, random_state=seed, **config['cifar-10'])

    for i in range(n_seeds):
        np.random.seed(i)
        torch.manual_seed(i)
        random.seed(i)

        out_path_i = os.path.join(out_dir, str(i))

        pretrain_network_dataset([hash], nasbench, dataset, device=device, dir_path=out_path_i,
                                 **config['pretrain'])


if __name__ == "__main__":
    main()
