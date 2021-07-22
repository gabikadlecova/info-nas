from copy import copy

import click
import os

import numpy as np
import scipy
import scipy.stats
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import torch
from matplotlib import pyplot as plt

from info_nas.datasets.io.create_dataset import load_io_dataset


def fit_eval_gp(train_features, train_accuracies, test_features, regr_name):
    if regr_name == 'gp':
        regr = GaussianProcessRegressor()
    elif regr_name == 'svr':
        regr = SVR()
    elif regr_name == 'rf':
        regr = RandomForestRegressor()
    elif regr_name == 'rf1000':
        regr = RandomForestRegressor(n_estimators=1000)
    elif regr_name == 'gb':
        regr = GradientBoostingRegressor()

    regr.fit(train_features, train_accuracies)
    pred = regr.predict(test_features)

    return pred


def plot_acc(test_target_acc, test_pred_acc, acc_map, title, save_path):
    test_target_acc = test_target_acc[acc_map]
    test_pred_acc = test_pred_acc[acc_map]

    pearson_r = scipy.stats.pearsonr(test_target_acc, test_pred_acc)
    rmse = sklearn.metrics.mean_squared_error(test_target_acc, test_pred_acc, squared=False)
    print(f"RMSE: {rmse}, Pearson's r: {pearson_r}")

    with open(os.path.join(save_path, f'{title}_metrics.txt'), 'w+') as f:
        f.write(f'RMSE: {rmse}\n')
        f.write(f'Pearson\'s r: {pearson_r}\n')

    plt.figure()
    bins = np.linspace(0.8, 1, 301)
    plt.plot([0.8, 1], [0.8, 1], 'yellowgreen', linewidth=2)
    H, xedges, yedges = np.histogram2d(test_target_acc, test_pred_acc, bins=bins)

    H = H.T
    Hm = np.ma.masked_where(H < 1, H)
    X, Y = np.meshgrid(xedges, yedges)
    palette = copy(plt.cm.viridis)
    palette.set_bad('w', 1.0)
    plt.pcolormesh(X, Y, Hm, cmap=palette)

    plt.xlabel('Test Accuracy')
    plt.ylabel('Predicted Accuracy')
    plt.yticks(ticks=[0.8, 0.85, 0.90, 0.95])
    plt.xticks(ticks=[0.8, 0.85, 0.9])
    plt.xlim(0.8, 0.95)
    plt.ylim(0.8, 0.95)

    plt.title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{title}.png"))
    #plt.show()


@click.command()
@click.argument('emb_path')
@click.option('--dir_name', default='.')
@click.option('--train_dataset', default='../data/train_long.pt')
@click.option('--regr_name', default='rf')
@click.option('--n_hashes', default=None, type=int)
@click.option('--use_train/--use_any', default=False)
@click.option('--seed', default=1)
def main(emb_path, dir_name, train_dataset, regr_name, n_hashes, use_train, seed):
    np.random.seed(seed)

    f_path = os.path.join(dir_name, emb_path)
    print("load arch2vec from: {}".format(f_path))

    hashes = []
    features = []
    val_acc = []
    test_acc = []
    embedding = torch.load(f_path)

    for ind in range(len(embedding)):
        hashes.append(embedding[ind]['hash'])
        features.append(embedding[ind]['feature'])
        val_acc.append(embedding[ind]['valid_accuracy'])
        test_acc.append(embedding[ind]['test_accuracy'])

    features = torch.stack(features)
    print('Loading finished. pretrained embeddings shape: {}'.format(features.shape))

    hashes = np.array(hashes)
    features = np.array(features)
    val_acc = np.array(val_acc)
    test_acc = np.array(test_acc)

    print("Loading train dataset...")
    dataset = load_io_dataset(train_dataset)
    unique_train_hashes = np.unique(dataset['net_hashes'])

    if use_train and n_hashes is not None:
        selected_hashes = np.random.choice(unique_train_hashes, n_hashes, replace=False)
    elif use_train:
        selected_hashes = unique_train_hashes
    else:
        assert n_hashes is not None, "If selecting random nets from the whole dataset, n_hashes must be specified."
        unique_hashes = np.unique(hashes)
        unique_hashes = unique_hashes[~np.in1d(unique_hashes, unique_train_hashes)]
        selected_hashes = np.random.choice(unique_hashes, n_hashes, replace=False)

    print(f"Selected {'all' if n_hashes is not None else n_hashes} {'train' if use_train else 'random'} "
          f"hashes for fit.")

    train_map = np.in1d(hashes, selected_hashes)

    train_features = features[train_map]
    train_val_acc = val_acc[train_map]
    train_test_acc = test_acc[train_map]

    print(train_features.shape, train_val_acc.shape, train_test_acc.shape)

    test_features = features[~train_map]
    test_val_acc = val_acc[~train_map]
    test_test_acc = test_acc[~train_map]

    train_y = [train_val_acc, train_test_acc] * 2
    x = [features, test_features] * 2
    y = [val_acc, test_val_acc, test_acc, test_test_acc]
    titles = ["All features - validation accuracy", "Test features - validation accuracy",
              "All features - test accuracy", "Test features - test accuracy"]

    save_dir = os.path.join(dir_name, f'{regr_name}_{n_hashes}_{seed}_{emb_path}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for eval_features, eval_acc, train_acc, title in zip(x, y, train_y, titles):
        preds = fit_eval_gp(train_features, train_acc, eval_features, regr_name)
        print(title)
        print('---------------------')

        plot_map = eval_acc > 0.8
        plot_acc(eval_acc, preds, plot_map, title, save_dir)


if __name__ == "__main__":
    main()

# todo nezapomenout spustit ABLATIONS
