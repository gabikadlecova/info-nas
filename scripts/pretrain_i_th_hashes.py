import click
import json
import os
import pandas as pd
import torch

from info_nas.config import local_dataset_cfg, load_json_cfg
from info_nas.datasets.networks.pretrained import pretrain_network_dataset
from nasbench import api
from nasbench_pytorch.datasets.cifar10 import prepare_dataset
from scripts.utils import mkdir_if_not_exists


@click.command()
@click.argument('hashes_dir')
@click.argument('chunk_no')
@click.option('--prefix', default='hashes_')
@click.option('--nasbench_path', default='../data/nasbench_only108.tfrecord')
@click.option('--config_path', default='../configs/pretrain_config.json')
@click.option('--root', default='../data/cifar/')
@click.option('--seed', default=1)
@click.option('--device', default='cuda')
def main(hashes_dir, chunk_no, prefix, nasbench_path, config_path, root, seed, device):
    device = torch.device(device)

    # load hashes

    chunk_path = os.path.join(hashes_dir, f"{prefix}{chunk_no}.csv")
    df = pd.read_csv(chunk_path)
    hash_list = df['hashes'].to_list()

    out_dir = os.path.join(hashes_dir, f"out_{chunk_no}/")
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
    dataset = prepare_dataset(root=root, random_state=seed, **config['cifar-10'])

    pretrain_network_dataset(hash_list, nasbench, dataset, device=device, dir_path=out_dir,
                             **config['pretrain'])


if __name__ == "__main__":
    main()
