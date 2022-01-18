import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torchvision
from info_nas.datasets.io.transforms import IncludeBias, MultByWeights, SortByWeights
from info_nas.datasets.arch2vec_dataset import prepare_labeled_dataset


def get_pred_and_orig(gen, model=None, print_freq=1000):
    orig = []
    pred = []
    info = []
    weights = []
    labels = []


    for i, batch in enumerate(gen):
        if i % print_freq == 0:
            print(f"Batch {i}")

        info.append({w: batch[w] for w in ['label', 'hash', 'ref_id']})

        b = batch['adj'], batch['ops'], batch['input'], batch['output']

        if model is not None:
            res = model(b[1].to(device), b[0].to(device), b[2].to(device))
            pred.append(res[-1].detach().cpu().numpy())
        orig.append(b[3].numpy())
        weights.append(np.concatenate([batch['weights'], batch['bias'][:, :, np.newaxis]], axis=-1))
        labels.append(batch['label'].numpy())

    orig = np.vstack(orig)
    weights = np.vstack(weights)
    labels = np.hstack(labels)
    
    if model is None:
        return orig, info, weights, labels
    
    pred = np.vstack(pred)
    return orig, pred, info, weights, labels



def get_labeled_data(data_pt, nasbench, nb_dataset, input_dataset, top_k=None):
    transforms = [IncludeBias(), MultByWeights(include_bias=True), SortByWeights(return_top_n=top_k)]
    transforms = torchvision.transforms.Compose(transforms)


    data, _ = prepare_labeled_dataset(data_pt, nasbench, device=torch.device('cpu'),
                                      nb_dataset=nb_dataset, dataset=input_dataset, remove_labeled=False)

    labeled = labeled_network_dataset(data, transforms=transforms, return_ref_id=True)
    return torch.utils.data.DataLoader(labeled, batch_size=32, shuffle=False, num_workers=4)


def normalize_feats(features, by_row=False, div_by_sigma=False, sub_mean=False, top_k=None):
    if by_row:
        mu = np.mean(features, axis=1)[:, np.newaxis]
        sigma = np.std(features, axis=1)[:, np.newaxis]
    else:
        mu = np.mean(features)
        sigma = np.std(features)

    features = features if not sub_mean else (features - mu)
    features = features if not div_by_sigma else (features / sigma)

    if top_k is not None:
        features = features[:, :top_k]
    return features


def plot_single_heatmap(features, sub_mean=False, div_by_sigma=False, by_row=False, top_k=None, **kwargs):
    features = normalize_feats(features, sub_mean=sub_mean, div_by_sigma=div_by_sigma, by_row=by_row, top_k=top_k)

    sns.heatmap(features, **kwargs)
    return features


def heatmap_diff(f_1, f_2, sub_mean=False, div_by_sigma=False, by_row=False, top_k=None,
                 use_abs=False, use_sq=False, plot_it=True, **kwargs):
    
    f_1 = normalize_feats(f_1, sub_mean=sub_mean, div_by_sigma=div_by_sigma, by_row=by_row, top_k=top_k)
    f_2 = normalize_feats(f_2, sub_mean=sub_mean, div_by_sigma=div_by_sigma, by_row=by_row, top_k=top_k)

    diff = f_1 - f_2
    diff = np.abs(diff) if use_abs else diff
    diff = np.square(diff) if use_sq else diff

    if plot_it:
        sns.heatmap(diff, **kwargs)
    return diff


def plot_hist(features, top_k=None, **kwargs):
    features = features if top_k is None else features[:, :top_k]

    sns.histplot(features, **kwargs)

