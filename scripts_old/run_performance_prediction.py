from copy import copy
from functools import partial

import click
import os

import numpy as np
import scipy
import scipy.stats
import sklearn
from arch2vec.extensions.get_nasbench101_model import get_arch2vec_model
from arch2vec.models.configs import configs
from arch2vec.utils import load_json, preprocessing
from info_nas.models.accuracy_model import AccuracyModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import torch
from matplotlib import pyplot as plt

from _old.datasets.io.create_dataset import load_io_dataset
from info_nas.models.utils import load_extended_vae

regressor_names = {
    'gp': GaussianProcessRegressor,
    'gp_norm': partial(GaussianProcessRegressor, normalize_y=True),
    'svr': SVR,
    'rf': RandomForestRegressor,
    'rf1000': partial(RandomForestRegressor, n_estimators=1000),
    'rf_tune': partial(RandomForestRegressor, n_estimators=200, max_features=4),
    'gb': GradientBoostingRegressor
}


def get_adj_ops(nb_dataset, hashes):
    nb_dataset = load_json(nb_dataset)
    hashmap = {j['hash']: i for i, j in nb_dataset.items()}
    chosen_inds = [hashmap[h] for h in hashes]

    get_it = lambda idx, what: torch.Tensor(nb_dataset[str(idx)][what]).unsqueeze(0).cuda()
    adj_ops = [(get_it(i, 'module_adjacency'), get_it(i, 'module_operations')) for i in chosen_inds]
    return adj_ops


def pred_with_net(model_path, adj_ops, config=4, device=None, print_freq=1000, batch_size=256):
    vae_model, _ = get_arch2vec_model(device=device)
    args = [vae_model]
    model, _ = load_extended_vae(model_path, args, device=device, daclass=AccuracyModel)
    cfg = configs[config]

    batch_adj = []
    batch_ops = []
    res = []
    with torch.no_grad():
        for i, (adj, ops) in enumerate(adj_ops):
            if i % print_freq == 0:
                print(i)

            batch_adj.append(adj)
            batch_ops.append(ops)
            if len(batch_adj) < batch_size and i != len(adj_ops) - 1:
                continue

            # construct batch
            adj, ops = torch.vstack(batch_adj), torch.vstack(batch_ops)
            batch_adj = []
            batch_ops = []

            adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
            _, _, _, _, _, acc = model(ops, adj)
            res.append(acc.detach().cpu().numpy())

    return np.hstack(res).flatten()


def fit_eval_gp(train_features, train_accuracies, test_features, regr_name, **kwargs):
    if regr_name in regressor_names:
        regr = regressor_names[regr_name]
        regr = regr() if regr_name != 'rf' else regr(**kwargs)
    else:
        raise ValueError("Unsupported regressor name, supported: ")

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
@click.option('--save_dir', default=None)
@click.option('--train_dataset', default='../data/train_long.pt',
              help="Load the train set to get unique hashes. The regressor is fitted either on train nets, or on all "
                   "other nets.")
@click.option('--regr_name', default='rf',
              help="Name of the regressor, possible values: [rf, rf1000, gp, svr, gb], meaning:"
                   "Random forest, random forest with 1000 estimators, gaussian process, support vector regressor,"
                   " gradient boosting.")
@click.option('--max_features', default="auto")
@click.option('--n_estimators', default=100)
@click.option('--n_hashes', default=None, type=int, help="Number of hashes to select for the fit.")
@click.option('--use_train/--use_any', default=False,
              help="If True, use train hashes, if False, use unseen networks (from the rest of the NAS-Bench-101).")
@click.option('--seed', default=1, help="Seed to use.")
@click.option('--model_path', default=None, help="Use net as a performance predictor instead.")
@click.option('--nb_dataset', default='../data/nb_dataset.json', help="Arch2vec nasbench dataset save path.")
@click.option('--device', default=None)
def main(emb_path, save_dir, train_dataset, regr_name, max_features, n_estimators, n_hashes, use_train, seed,
         model_path, nb_dataset, device):
    """
    Fit and evaluate a performance predictior using the embedded features.

    Args: EMB_PATH - Name of the saved features (generated in the arch2vec repository).
    """
    np.random.seed(seed)
    device = device if device is None else torch.device(device)

    print("load arch2vec from: {}".format(emb_path))
    dir_name, emb_base = os.path.split(emb_path)

    hashes = []
    features = []
    val_acc = []
    test_acc = []
    embedding = torch.load(emb_path)

    for ind in range(len(embedding)):
        hashes.append(embedding[ind]['hash'])
        features.append(embedding[ind]['feature'].flatten())
        val_acc.append(embedding[ind]['valid_accuracy'])
        test_acc.append(embedding[ind]['test_accuracy'])

    features = torch.stack(features)
    #if 'info' in emb_base:
    #    features = features[:, 0]
    print('Loading finished. pretrained embeddings shape: {}'.format(features.shape))

    hashes = np.array(hashes)
    features = np.array(features)
    val_acc = np.array(val_acc)
    test_acc = np.array(test_acc)

    print("Loading train dataset...")
    dataset = load_io_dataset(train_dataset)
    unique_train_hashes = np.unique(dataset['net_hashes'])

    def select_outside_train(n):
        unique_hashes = np.unique(hashes)
        unique_hashes = unique_hashes[~np.in1d(unique_hashes, unique_train_hashes)]
        return np.random.choice(unique_hashes, n, replace=False)

    if use_train and n_hashes is not None:
        if n_hashes < len(unique_train_hashes):
            selected_hashes = np.random.choice(unique_train_hashes, n_hashes, replace=False)
        elif n_hashes == len(unique_train_hashes):
            selected_hashes = unique_train_hashes
        else:
            selected_hashes = select_outside_train(n_hashes - len(unique_train_hashes))
            selected_hashes = np.hstack([selected_hashes, unique_train_hashes])
    elif use_train:
        selected_hashes = unique_train_hashes
    else:
        assert n_hashes is not None, "If selecting random nets from the whole dataset, n_hashes must be specified."
        selected_hashes = select_outside_train(n_hashes)

    print(f"Selected {'all' if n_hashes is not None else n_hashes} {'train' if use_train else 'random'} "
          f"hashes for fit.")

    train_map = np.in1d(hashes, selected_hashes)

    train_features = features[train_map]
    train_val_acc = val_acc[train_map]
    train_test_acc = test_acc[train_map]

    print(train_features.shape, train_val_acc.shape, train_test_acc.shape)

    test_hashes = hashes[~train_map]
    test_features = features[~train_map]
    test_val_acc = val_acc[~train_map]
    test_test_acc = test_acc[~train_map]

    train_y = [train_val_acc, train_test_acc] * 2
    x = [features, test_features] * 2
    y = [val_acc, test_val_acc, test_acc, test_test_acc]
    titles = ["All features - validation accuracy", "Test features - validation accuracy",
              "All features - test accuracy", "Test features - test accuracy"]
    hash_list = [hashes, test_hashes] * 2

    if save_dir is None:
        save_dir = dir_name

    emb_name = os.path.basename(emb_path)
    save_dir = os.path.join(save_dir, f'{regr_name}_{n_hashes}_{seed}_{"train" if use_train else "any"}_{emb_name}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if model_path is not None:
        model_name = os.path.basename(model_path)
        model_save_dir = os.path.join(
            save_dir, f'model-prediction_{n_hashes}_{seed}_{"train" if use_train else "any"}_{model_name}'
        )

    for eval_features, eval_acc, train_acc, title, eval_hashes in zip(x, y, train_y, titles, hash_list):
        #if 'Test features' in title:
        #    continue
        print(title)
        print('---------------------')
        plot_map = eval_acc > 0.8

        # use model to predict
        if model_path is not None:
            adj_ops = get_adj_ops(nb_dataset, eval_hashes)
            preds = pred_with_net(model_path, adj_ops, device=device)
            plot_acc(eval_acc, preds, plot_map, title, model_save_dir)

        preds = fit_eval_gp(train_features, train_acc, eval_features, regr_name,
                            max_features=max_features, n_estimators=n_estimators)
        plot_acc(eval_acc, preds, plot_map, title, save_dir)


if __name__ == "__main__":
    main()
