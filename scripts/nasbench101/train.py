import click
import pandas as pd
import pytorch_lightning as pl

from info_nas.datasets.io_dataset import IODataModule, IOData
from info_nas.datasets.search_spaces.nasbench101 import Nasbench101Data
from info_nas.datasets.transforms import MultiplyByWeights, SortByWeights, IncludeBias
from info_nas.models.vae.arch2vec import Arch2vecPreprocessor

from info_nas.models.io_model import ConcatConvModel
from info_nas.models.vae.arch2vec import Arch2vecModel

from torch.nn import MSELoss
from info_nas.metrics.arch2vec import VAELoss, ReconstructionMetrics, ValidityUniqueness

from searchspace_train.datasets.nasbench101 import load_nasbench
from torchvision.transforms import Compose

from info_nas.trainer import InfoNAS


def load_hashes(path):
    data = pd.read_csv(path)
    return data['hashes']


def get_preprocessor():
    return Arch2vecPreprocessor()


def get_label_transforms():
    return Compose([IncludeBias(), MultiplyByWeights(), SortByWeights()])


def get_metrics(prepro, vae):
    return {'vae': ReconstructionMetrics(prepro), 'vu': ValidityUniqueness(prepro, vae)}


@click.command()
@click.option('--nb', default='../nasbench_only108.tfrecord')
@click.option('--train_hashes', required=True)
@click.option('--val_hashes', required=True)
@click.option('--io_path', required=True)
def train(nb, train_hashes, val_hashes, io_path):
    nb = load_nasbench(nb)
    prepro = get_preprocessor()

    train_hashes, val_hashes = load_hashes(train_hashes), load_hashes(val_hashes)

    # data
    network_data = Nasbench101Data(nb, prepro)
    io_data = IOData(load_path=io_path)
    transform = get_label_transforms()

    dm = IODataModule(network_data, io_data, label_transform=transform,
                      train_hash_list=train_hashes, val_hash_list=val_hashes)

    # model
    vae = Arch2vecModel()
    model = ConcatConvModel(vae, 3, 513)

    # loss and metrics
    loss = VAELoss()
    labeled_loss = MSELoss()

    train_metrics = get_metrics(prepro, vae)
    val_metrics = get_metrics(prepro, vae)

    # trainer
    pl_trainer = pl.Trainer()
    infonas = InfoNAS(model, loss, labeled_loss, prepro, train_metrics=train_metrics, valid_metrics=val_metrics)

    pl_trainer.fit(infonas, datamodule=dm)


if __name__ == "__main__":
    train()