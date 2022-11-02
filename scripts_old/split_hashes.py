import os

import click
import pandas as pd

from scripts_old.utils import mkdir_if_not_exists


@click.command()
@click.argument('source_hashes')
@click.argument('save_dir', default='../data/hashes/')
@click.option('--n_splits', default=60)
def main(source_hashes, save_dir, n_splits):
    hashes = pd.read_csv(source_hashes)

    if len(hashes) % n_splits == 0:
        div = n_splits
    else:
        div = n_splits - 1

    chunk_size = len(hashes) // div
    chunk_dfs = [hashes[(i * chunk_size):((i + 1) * chunk_size)] for i in range(n_splits)]

    mkdir_if_not_exists(save_dir)
    chunks_dir = os.path.join(save_dir, f'{os.path.basename(source_hashes[:-4])}_{n_splits}_splits/')
    mkdir_if_not_exists(chunks_dir)

    for i, df in enumerate(chunk_dfs):
        save_path = os.path.join(chunks_dir, f'hashes_{i}.csv')
        df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
