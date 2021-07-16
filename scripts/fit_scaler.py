import click
import os

import numpy as np

from info_nas.datasets.io.transforms import Scaler
from info_nas.datasets.io.create_dataset import load_io_dataset


@click.command()
@click.argument('scale_name')
@click.option('--dataset', default='../data/train_labeled.pt')
@click.option('--scale_save_dir', default='../data/scales/')
@click.option('--per_label/--per_net', default=False)
@click.option('--weighted/--original', default=False)
@click.option('--axis', default=None, type=int)
@click.option('--include_bias/--no_bias', default=True)
def main(scale_name, dataset, scale_save_dir, per_label, weighted, axis, include_bias):
    dataset = load_io_dataset(dataset)

    if not os.path.exists(scale_save_dir):
        os.mkdir(scale_save_dir)

    scale_save_path = os.path.join(scale_save_dir,
                                   f"scale-{scale_name}"
                                   f"{'-include_bias' if include_bias else ''}"
                                   f"{'-per_label' if per_label else ''}"
                                   f"{'-weighted' if weighted else ''}"
                                   f"{'-axis_' + str(axis) if axis is not None else ''}.pickle")

    outputs = dataset['outputs'].numpy()
    if include_bias:
        one_vec = np.ones((len(outputs), 1))
        outputs = np.hstack([outputs, one_vec])

    scale = Scaler(per_label=per_label, axis=axis, weighted=weighted, include_bias=include_bias)
    scale.fit(outputs,
              dataset['net_hashes'],
              labels=dataset['labels'][dataset['inputs']].numpy(),
              net_repo=dataset['net_repo'],
              save_path=scale_save_path)


if __name__ == "__main__":
    main()
