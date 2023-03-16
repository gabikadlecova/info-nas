import os.path

import click
import pytorch_lightning as pl

from info_nas.datasets.base import NetworkDataModule, NetworkDataset
from info_nas.datasets.io_dataset import IOData, IODataset
from info_nas.datasets.search_spaces.nasbench101 import Nasbench101Data
from info_nas.datasets.transforms import MultiplyByWeights, SortByWeights, IncludeBias
from info_nas.models.vae.arch2vec import Arch2vecPreprocessor

from info_nas.models.io_model import ConcatConvModel
from info_nas.models.vae.arch2vec import Arch2vecModel

from torch.nn import MSELoss
from info_nas.metrics.arch2vec import VAELoss, ReconstructionMetrics, ValidityUniqueness, ValidityNasbench101

from searchspace_train.datasets.nasbench101 import load_nasbench
from torchvision.transforms import Compose

from info_nas.trainer import InfoNAS, save_to_trainer_path


def get_preprocessor():
    return Arch2vecPreprocessor()


def get_label_transforms():
    return Compose([IncludeBias(), MultiplyByWeights(), SortByWeights()])


def get_metrics(prepro, vae, nb):
    return {
        'vae': ReconstructionMetrics(prepro),
        'vu': ValidityUniqueness(prepro, vae, ValidityNasbench101(nb))
    }


def _base_dirs(base, *args):
    return [os.path.join(base, p) for p in args]


@click.command()
@click.option('--base_dir', default='.')
@click.option('--nb', default='../nasbench_only108.tfrecord')
@click.option('--train_hashes', required=True)
@click.option('--val_hashes', required=True)
@click.option('--io_path', required=True)
def train(base_dir, nb, train_hashes, val_hashes, io_path):
    nb, train_hashes, val_hashes, io_path = _base_dirs(base_dir, nb, train_hashes, val_hashes, io_path)

    nb = load_nasbench(nb)
    prepro = get_preprocessor()

    # data
    network_data = Nasbench101Data(nb, prepro)
    io_data = IOData(load_path=io_path)
    transform = get_label_transforms()
    nb_hashes = [h for h in nb.hash_iterator()]

    unlabeled_train = NetworkDataset(nb_hashes, network_data)
    labeled_train = IODataset(train_hashes, network_data, io_data, label_transform=transform)

    labeled_val = IODataset(val_hashes, network_data, io_data, label_transform=transform)

    dm = NetworkDataModule(network_data, {'labeled': labeled_train, 'unlabeled': unlabeled_train},
                           val_datasets=labeled_val)

    # model
    vae = Arch2vecModel()
    model = ConcatConvModel(3, 513, start_channels=32)

    # loss and metrics
    loss = VAELoss()
    labeled_loss = MSELoss()

    train_metrics = get_metrics(prepro, vae, nb)
    val_metrics = get_metrics(prepro, vae, nb)

    # trainer
    pl_trainer = pl.Trainer(max_epochs=2)
    infonas = InfoNAS(vae, model, loss, labeled_loss, prepro, train_metrics=train_metrics, valid_metrics=val_metrics)
    save_to_trainer_path(pl_trainer, infonas)

    pl_trainer.fit(infonas, datamodule=dm)


if __name__ == "__main__":
    train()
