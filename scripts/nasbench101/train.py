import json
import os.path
from functools import partial

import click
import pytorch_lightning as pl

from info_nas.config import INFONAS_CONFIGS
from info_nas.datasets.base import NetworkDataModule, NetworkDataset
from info_nas.datasets.io_dataset import load_from_config, get_n_dataloaders

from info_nas.models.io_model import ConcatConvModel
from info_nas.models.vae.arch2vec import Arch2vecModel

from searchspace_train.datasets.nasbench101 import load_nasbench

from info_nas.vae_models import InfoNAS, save_to_trainer_path


def init_from_scratch(config, nb, n_val):
    vae = Arch2vecModel()
    model = ConcatConvModel(3, 513, start_channels=32)

    config_func = INFONAS_CONFIGS[config]
    config = config_func(vae, nb, n_val=n_val, labeled_model=model)

    return config, InfoNAS(vae, model, config['loss'], config['labeled_loss'], config['preprocessor'],
                           metrics=config['metrics'])


def init_from_checkpoint(config, nb, n_val, ckpt_dir, weights_name):
    config_func = INFONAS_CONFIGS[config]
    return InfoNAS.load_from_checkpoint_dir(ckpt_dir, weights_name, config_func, n_val=n_val, nb=nb)


@click.command()
@click.option('--base_dir', default='.')
@click.option('--nb', default='../nasbench_only108.tfrecord')
@click.option('--config', default='arch2vec_nasbench101')
@click.option('--data_config', required=True)
@click.option('--epochs', default=5)
@click.option('--ckpt_dir', default=None)
@click.option('--weights_name', default=None)
def train(base_dir, nb, config, data_config, epochs, ckpt_dir, weights_name):
    print("Loading nasbench...")
    nb = os.path.join(base_dir, nb)
    nb = load_nasbench(nb)  # TODO more general load (e.g. move to cfg)
    print("Nasbench loaded")

    with open(data_config, 'r') as f:
        data_config = json.load(f)
    n_val = get_n_dataloaders(data_config, 'val')

    # model
    if ckpt_dir is None:
        config, infonas = init_from_scratch(config, nb, n_val)
    else:
        config, infonas = init_from_checkpoint(config, nb, n_val, ckpt_dir, weights_name)

    # datasets
    network_data = config['network_data']
    io_datasets = load_from_config(data_config, network_data, base_dir=base_dir)

    unlabeled_train = NetworkDataset(network_data.get_hashes(), network_data)
    labeled_train = io_datasets['train']
    labeled_val = io_datasets['val']

    dm = NetworkDataModule(network_data, {'labeled': labeled_train, 'unlabeled': unlabeled_train},
                           val_datasets=labeled_val)

    # trainer
    pl_trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=1)
    save_to_trainer_path(pl_trainer, infonas)

    ckpt_path = None if ckpt_dir is None else os.path.join(ckpt_dir, 'checkpoints', weights_name)
    pl_trainer.fit(infonas, datamodule=dm, ckpt_path=ckpt_path)

    print()


if __name__ == "__main__":
    train()
