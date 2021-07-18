import datetime
import json
import os
import pickle

import click
import torch

from info_nas.config import load_json_cfg, local_model_cfg
from info_nas.datasets.io.transforms import get_transforms, get_all_scales

from info_nas.datasets.arch2vec_dataset import get_labeled_unlabeled_datasets
from nasbench import api

from info_nas.trainer import train


# @click.option('--nasbench_path', default='../data/nasbench_only108.tfrecord')

@click.command()
@click.option('--train_path', default='../data/train_long.pt')
@click.option('--valid_path', default='../data/valid_long.pt')
@click.option('--unseen_valid_path', default='../data/test_train_long.pt')
@click.option('--scale_dir', default='../data/scales/')
@click.option('--checkpoint_path', default='../data/vae_checkpoints/')
@click.option('--nasbench_path', default='../data/nasbench.pickle')
@click.option('--model_cfg', default=None)
@click.option('--use_ref/--no_ref', default=False)
@click.option('--test_is_splitted/--split_test', default=False)
@click.option('--use_unseen_data/--no_unseen_data', default=False)
@click.option('--device', default='cuda')
@click.option('--seed', default=1)
@click.option('--batch_size', default=32)
@click.option('--epochs', default=7)
def run(train_path, valid_path, unseen_valid_path, scale_dir, checkpoint_path, nasbench_path, model_cfg, use_ref,
        test_is_splitted, use_unseen_data, device, seed, batch_size, epochs):

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
    labeled, unlabeled = get_labeled_unlabeled_datasets(nb, device=device, seed=1,
                                                        train_pretrained=None,
                                                        valid_pretrained=None,
                                                        train_labeled_path=train_path,
                                                        valid_labeled_path=valid_path,
                                                        test_labeled_train_path=unseen_valid_path,
                                                        test_valid_split=None if test_is_splitted else 0.1)

    # load all scaling
    scale_config = model_cfg["scale"]
    include_bias = scale_config["include_bias"]
    normalize = scale_config["normalize"]
    mult_by_weights = scale_config["multiply_by_weights"]
    use_scale_whole = scale_config["scale_whole"]

    scale_train, scale_valid, scale_whole = get_all_scales(scale_dir, scale_config)
    scale_whole = scale_whole if use_scale_whole else None
    print(f"Scale paths: {scale_train}, {scale_valid}, {scale_whole}")

    transforms = get_transforms(scale_train, include_bias, normalize, mult_by_weights, scale_whole_path=scale_whole)
    val_transforms = get_transforms(scale_valid, include_bias, normalize, mult_by_weights, scale_whole_path=scale_whole)

    timestamp = datetime.datetime.now().strftime('%Y-%d-%m_%H-%M-%S')
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    checkpoint_path = os.path.join(checkpoint_path, timestamp)
    os.mkdir(checkpoint_path)

    # save config
    config_path = os.path.join(checkpoint_path, 'config.json')
    with open(config_path, 'w+') as f:
        json.dump(model_cfg, f, indent=4)

    model, metrics, loss = train(labeled, unlabeled, nb, transforms=transforms, valid_transforms=val_transforms,
                                 checkpoint_dir=checkpoint_path, device=device, use_reference_model=use_ref,
                                 batch_len_labeled=4, model_config=model_cfg,
                                 batch_size=batch_size, seed=seed, epochs=epochs)

    with open(os.path.join(checkpoint_path, 'metrics.pickle'), 'wb') as f:
        pickle.dump(metrics, f)

    with open(os.path.join(checkpoint_path, 'losses.pickle'), 'wb') as f:
        pickle.dump(loss, f)

    # TODO deterministic etc


if __name__ == "__main__":
    run()
