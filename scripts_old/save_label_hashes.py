import click
import os
import pandas as pd

from info_nas.config import local_dataset_cfg, load_json_cfg
from _old.datasets import split_to_labeled, generate_or_load_nb_dataset
from nasbench import api

from scripts_old.utils import mkdir_if_not_exists


@click.command()
@click.argument('save_dir', default='../data/hashes/')
@click.option('--nasbench_path', default='../data/nasbench_only108.tfrecord')
@click.option('--arch2vec_path', default='../data/nb_dataset.json')
@click.option('--seed', default=1)
@click.option('--config_path', default='../configs/pretrain_config.json')
@click.option('--percent_labeled', default=0.01)
def main(save_dir, nasbench_path, arch2vec_path, seed, config_path, percent_labeled):
    if not len(config_path) or config_path is None:
        config = local_dataset_cfg
    else:
        config = load_json_cfg(config_path)

    nasbench = api.NASBench(nasbench_path)

    # get random hashes
    nb_dataset = generate_or_load_nb_dataset(nasbench, save_path=arch2vec_path, seed=seed, batch_size=None,
                                             val_batch_size=None, **config['nb_dataset'])
    train_hashes, valid_hashes = split_to_labeled(nb_dataset, seed=seed, percent_labeled=percent_labeled)

    # save hashes
    mkdir_if_not_exists(save_dir)
    source_dir = os.path.join(save_dir, 'source_hashes/')
    mkdir_if_not_exists(source_dir)

    train_df = pd.DataFrame(data=train_hashes, columns=['hashes'])
    valid_df = pd.DataFrame(data=valid_hashes, columns=['hashes'])

    train_df.to_csv(os.path.join(source_dir, 'train_hashes.csv'), index=False)
    valid_df.to_csv(os.path.join(source_dir, 'valid_hashes.csv'), index=False)


if __name__ == "__main__":
    main()
