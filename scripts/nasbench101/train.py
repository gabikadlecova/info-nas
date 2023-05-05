import json
import os.path

import click
import pytorch_lightning as pl

from info_nas.datasets.search_spaces.nasbench101 import load_nb_datasets
from info_nas.config import INFONAS_CONFIGS
from info_nas.datasets.base import NetworkDataModule, join_dataset_iterables
from info_nas.datasets.io_dataset import get_n_dataloaders, load_io_from_config

from info_nas.models.io_model import ConcatConvModel
from info_nas.models.vae.arch2vec import Arch2vecModel

from searchspace_train.datasets.nasbench101 import load_nasbench

from info_nas.vae_models import InfoNAS, save_to_trainer_path, NetworkVAE


def init_from_scratch(config, nb, n_val, unlabeled=False, debug=False):
    vae = Arch2vecModel()
    model = None if unlabeled else ConcatConvModel(3, 513, start_channels=32)

    config_func = INFONAS_CONFIGS[config]
    config = config_func(vae, nb, n_val=n_val, labeled_model=model, debug=debug)

    if unlabeled:
        return config, NetworkVAE(vae, config['loss'], config['preprocessor'], metrics=config['metrics'])

    return config, InfoNAS(vae, model, config['loss'], config['labeled_loss'], config['preprocessor'],
                           metrics=config['metrics'])


def init_from_checkpoint(config, nb, n_val, ckpt_dir, weights_name, unlabeled=False, debug=False):
    config_func = INFONAS_CONFIGS[config]
    cls = NetworkVAE if unlabeled else InfoNAS
    return cls.load_from_checkpoint_dir(ckpt_dir, weights_name, config_func, n_val=n_val, nb=nb, debug=debug)


@click.command()
@click.option('--base_dir', default='.')
@click.option('--nb', default='../nasbench_only108.tfrecord')
@click.option('--config', default='arch2vec_nasbench101')
@click.option('--data_config', required=True)
@click.option('--epochs', default=5)
@click.option('--ckpt_dir', default=None)
@click.option('--weights_name', default=None)
@click.option('--debug/--no_debug', default=False)
@click.option('--unlabeled/--labeled', default=False, help="Labeled or unlabeled model.")
@click.option('--as_labeled/--as_unlabeled', default=False, help="If unlabeled model, determine the data mode.")
def train(base_dir, nb, config, data_config, epochs, ckpt_dir, weights_name, debug, unlabeled, as_labeled):
    print("Loading nasbench...")
    nb = os.path.join(base_dir, nb)
    nb = load_nasbench(nb)  # TODO more general load (e.g. move to cfg)
    print("Nasbench loaded")

    with open(data_config, 'r') as f:
        data_config = json.load(f)
    nval = get_n_dataloaders(data_config, 'val')

    # model
    if ckpt_dir is None:
        config, model = init_from_scratch(config, nb, nval, unlabeled=unlabeled, debug=debug)
    else:
        config, model = init_from_checkpoint(config, nb, nval, ckpt_dir, weights_name, unlabeled=unlabeled, debug=debug)

    # datasets
    network_data = config['network_data']
    unlabeled_datasets = load_nb_datasets(data_config['unlabeled'], network_data, base_dir=base_dir)
    train_set, val_set = unlabeled_datasets['train'], unlabeled_datasets['val']

    if not unlabeled or as_labeled:
        io_datasets = load_io_from_config(data_config['labeled'], network_data, base_dir=base_dir)
        train_set = {'labeled': io_datasets['train'], 'unlabeled': train_set}
        val_set = join_dataset_iterables(val_set, io_datasets['val'])

    dm = NetworkDataModule(network_data, train_set, val_datasets=val_set)

    # trainer
    debug_args = {}
    if debug:
        debug_args = {'log_every_n_steps': 1, 'limit_train_batches': 2, 'limit_val_batches': 2}

    pl_trainer = pl.Trainer(max_epochs=epochs, **debug_args)
    save_to_trainer_path(pl_trainer, model)

    ckpt_path = None if ckpt_dir is None else os.path.join(ckpt_dir, 'checkpoints', weights_name)
    pl_trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

    print()


if __name__ == "__main__":
    train()
