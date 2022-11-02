import datetime
import json
import os
import pickle

import click
import torch
from info_nas.models.accuracy_model import train_as_infonas

from info_nas.config import load_json_cfg, local_model_cfg

from _old.datasets import get_labeled_unlabeled_datasets
from nasbench import api

from _old.trainer_old import train


# @click.option('--nasbench_path', default='../data/nasbench_only108.tfrecord')
from scripts_old.utils import experiment_transforms


@click.command()
@click.option('--train_path', default='../data/train_long.pt', help="Path to the saved train IO dataset.")
@click.option('--valid_path', default='../data/valid_long.pt',
              help="Path to the saved validation (unseen networks) IO dataset.")
@click.option('--unseen_valid_path', default='../data/test_train_long.pt',
              help="Path to the saved validation (unseen images) IO dataset.")
@click.option('--checkpoint_path', default='../data/vae_checkpoints/',
              help="Checkpoint directory; the checkpoint is created in a subdirectory named by the current timestamp.")
@click.option('--nasbench_path', default='../data/nasbench.pickle',
              help="Path to the nasbench dataset (pickle or directly downloaded from the nasbench original repo "
                   "(where the download link is provided).")
@click.option('--nb_dataset', default='../data/nb_dataset.json',
              help="Path to stored arch2vec dataset (if it does not exist yet, it will be created and saved there).")
@click.option('--cifar', default='../data/cifar/',
              help="Path to preloaded CIFAR-10 dataset, if it was not preloaded yet, it will be preloaded there now.")
@click.option('--model_cfg', default=None,
              help="Path to model config, if None, info_nas.config.local_model_cfg is used.")
@click.option('--use_ref/--no_ref', default=False,
              help="If True, train the arch2vec model alongside the extended model for reference.")
@click.option('--test_is_splitted/--split_test', default=False,
              help="If True, use the whole file from unseen_valid_path, if False, split a small dataset "
                   "(ratio 0.1) from it.")
@click.option('--use_unseen_data/--no_unseen_data', default=True,
              help="If True, use the unseen image validation set, if False, do not use it.")
@click.option('--use_accuracy/--use_io_data', default=False,
              help="If True, run semi supervised accuracy prediction instead.")
@click.option('--deterministic/--deterministic_off', default=False,
              help="Set torch and cudnn deterministic.")
@click.option('--device', default='cuda', help="Device for the training.")
@click.option('--seed', default=1, help="Seed to use.")
@click.option('--batch_size', default=32, help="Batch size for both labeled and unlabeled batches.")
@click.option('--epochs', default=7, help="Number of training epochs.")
def run(train_path, valid_path, unseen_valid_path, checkpoint_path, nasbench_path, nb_dataset, cifar,
        model_cfg, use_ref, test_is_splitted, use_unseen_data, use_accuracy, deterministic, device,
        seed, batch_size, epochs):
    """
    Run the training of the info-NAS model.
    """

    # load datasets
    if nasbench_path.endswith('.pickle'):
        with open(nasbench_path, 'rb') as f:
            nb = pickle.load(f)
    else:
        nb = api.NASBench(nasbench_path)

    if model_cfg is not None:
        model_cfg = load_json_cfg(model_cfg)
    else:
        model_cfg = local_model_cfg

    device = torch.device(device)
    unseen_valid_path = unseen_valid_path if use_unseen_data else None

    # the dataset should have the same splits every time
    labeled, unlabeled = get_labeled_unlabeled_datasets(nb, device=device, seed=1, nb_dataset=nb_dataset,
                                                        dataset=cifar,
                                                        train_pretrained=None,
                                                        valid_pretrained=None,
                                                        train_labeled_path=train_path,
                                                        valid_labeled_path=valid_path,
                                                        test_labeled_train_path=unseen_valid_path,
                                                        test_valid_split=None if test_is_splitted else 0.1)

    transforms = experiment_transforms(model_cfg, use_accuracy=use_accuracy)
    val_transforms = experiment_transforms(model_cfg, use_accuracy=use_accuracy)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    checkpoint_path = os.path.join(checkpoint_path, f"{timestamp}_seed_{seed}_bs_{batch_size}")
    os.mkdir(checkpoint_path)

    # save config
    config_path = os.path.join(checkpoint_path, 'config.json')
    with open(config_path, 'w+') as f:
        json.dump(model_cfg, f, indent=4)

    if use_accuracy:
        model, metrics, loss = train_as_infonas(labeled, unlabeled, nb, transforms=transforms,
                                                valid_transforms=val_transforms,
                                                checkpoint_dir=checkpoint_path, device=device, model_config=model_cfg,
                                                batch_size=batch_size, seed=seed, epochs=epochs,
                                                torch_deterministic=deterministic, cudnn_deterministic=deterministic)
    else:
        model, metrics, loss = train(labeled, unlabeled, nb, transforms=transforms, valid_transforms=val_transforms,
                                     checkpoint_dir=checkpoint_path, device=device, use_reference_model=use_ref,
                                     batch_len_labeled=4, model_config=model_cfg,
                                     batch_size=batch_size, seed=seed, epochs=epochs,
                                     torch_deterministic=deterministic, cudnn_deterministic=deterministic)

    with open(os.path.join(checkpoint_path, 'metrics.pickle'), 'wb') as f:
        pickle.dump(metrics, f)

    with open(os.path.join(checkpoint_path, 'losses.pickle'), 'wb') as f:
        pickle.dump(loss, f)

    # TODO deterministic etc


if __name__ == "__main__":
    run()
