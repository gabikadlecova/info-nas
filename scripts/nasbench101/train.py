import os.path

import click
import pytorch_lightning as pl

from info_nas.config import INFONAS_CONFIGS
from info_nas.datasets.base import NetworkDataModule, NetworkDataset
from info_nas.datasets.io_dataset import IOData, IODataset
from info_nas.datasets.transforms import get_label_transforms

from info_nas.models.io_model import ConcatConvModel
from info_nas.models.vae.arch2vec import Arch2vecModel

from searchspace_train.datasets.nasbench101 import load_nasbench

from info_nas.models import InfoNAS, save_to_trainer_path


def _base_dirs(base, *args):
    return [os.path.join(base, p) for p in args]


@click.command()
@click.option('--base_dir', default='.')
@click.option('--nb', default='../nasbench_only108.tfrecord')
@click.option('--config', default='arch2vec_nasbench101')
@click.option('--train_hashes', required=True)
@click.option('--val_hashes', required=True)
@click.option('--io_path', required=True)
def train(base_dir, nb, config, train_hashes, val_hashes, io_path):
    nb, train_hashes, val_hashes, io_path = _base_dirs(base_dir, nb, train_hashes, val_hashes, io_path)
    nb = load_nasbench(nb)  # TODO more general load (e.g. move to cfg)

    # model
    vae = Arch2vecModel()
    model = ConcatConvModel(3, 513, start_channels=32)

    config_func = INFONAS_CONFIGS[config]
    config = config_func(vae, nb, labeled_model=model)

    infonas = InfoNAS(vae, model, config['loss'], config['labeled_loss'], config['preprocessor'],
                      train_metrics=config['train_metrics'], valid_metrics=config['valid_metrics'])

    # datasets
    network_data = config['network_data']
    io_data = IOData(load_path=io_path)
    transform = get_label_transforms()

    unlabeled_train = NetworkDataset(network_data.get_hashes(), network_data)
    labeled_train = IODataset(train_hashes, network_data, io_data, label_transform=transform)
    labeled_val = IODataset(val_hashes, network_data, io_data, label_transform=transform)

    dm = NetworkDataModule(network_data, {'labeled': labeled_train, 'unlabeled': unlabeled_train},
                           val_datasets=labeled_val)

    # trainer
    pl_trainer = pl.Trainer(max_epochs=2)
    save_to_trainer_path(pl_trainer, infonas)

    pl_trainer.fit(infonas, datamodule=dm)

    print()


if __name__ == "__main__":
    train()
