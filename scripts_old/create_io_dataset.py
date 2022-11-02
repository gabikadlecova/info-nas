import pickle

import click
import torch

from info_nas.config import local_dataset_cfg, load_json_cfg
from _old.datasets.io.create_dataset import dataset_from_pretrained
from nasbench import api
from nasbench_pytorch.datasets.cifar10 import prepare_dataset


@click.command()
@click.argument('train_paths')
@click.option('--save_path', required=True)
@click.option('--nasbench_path', default='../data/nasbench_only108.tfrecord')
@click.option('--config_path', default=None)
@click.option('--dataset', default='../data/cifar/')
@click.option('--seed', default=1)
@click.option('--device', default='cuda')
@click.option('--use_test_data/--use_validation_data', default=False, is_flag=True,
              help="If True, use cifar test data instead of validation.")
def main(train_paths, save_path, nasbench_path, config_path, dataset, seed, device, use_test_data):
    device = torch.device(device)

    # load datasets
    if nasbench_path.endswith('.pickle'):
        with open(nasbench_path, 'rb') as f:
            nb = pickle.load(f)
    else:
        nb = api.NASBench(nasbench_path)

    if not len(config_path) or config_path is None:
        config = local_dataset_cfg
    else:
        config = load_json_cfg(config_path)

    dataset = prepare_dataset(root=dataset, random_state=seed, no_valid_transform=False, **config['cifar-10'])
    train_paths = train_paths.split(',')

    dataset_from_pretrained(train_paths, nb, dataset, save_path, device=device, use_test_data=use_test_data,
                            **config['io'])


if __name__ == "__main__":
    main()
