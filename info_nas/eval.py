# TODO eval loss dict separate, eval models
import os

import numpy as np
import pandas as pd
import torch
from arch2vec.extensions.get_nasbench101_model import eval_validity_and_uniqueness, eval_validation_accuracy
from arch2vec.utils import preprocessing

from info_nas.datasets.io.semi_dataset import enumerate_validation_labeled


def _metrics_list(res_dict, key):
    return res_dict.setdefault(key, [])


def eval_vae_validation(mod, valid_set, res_dict, device, config, verbose=2, n_validation=None):
    val_stats = eval_validation_accuracy(
        mod, valid_set, config=config, device=device, n_validation=n_validation
    )

    stats_names = ['acc_ops_val', 'mean_corr_adj_val', 'mean_fal_pos_adj_val', 'acc_adj_val']
    for n, val in zip(stats_names, val_stats):
        _metrics_list(res_dict, n).append(val)

    if verbose > 1:
        print(
            'validation set: acc_ops:{0:.4f}, mean_corr_adj:{1:.4f}, mean_fal_pos_adj:{2:.4f}, acc_adj:{3:.4f}'.format(
                *val_stats
            )
        )


def eval_labeled_validation(model, validation, device, config, loss_labeled):
    losses = []

    for batch in validation:
        adj, ops, inputs, outputs = batch[:4]
        adj, ops = adj.to(device), ops.to(device)
        adj, ops, prep_reverse = preprocessing(adj, ops, **config['prep'])

        inputs, outputs = inputs.to(device), outputs.to(device)

        model_out = model(ops, adj.to(torch.long), inputs)

        assert len(model_out) == 6  # TODO could differ
        outs_recon = model_out[-1]

        labeled_out = loss_labeled(outs_recon, outputs)
        losses.append(labeled_out.detach().item())

    return np.mean(losses)


def eval_epoch(model, model_labeled, model_reference, metrics_res_dict, Z, losses_total, losses_epoch, epoch, device,
               nasbench, valid_unlabeled, valid_labeled, config, loss_labeled, verbose=2):

    model_map = {
        'labeled': model_labeled,
        'unlabeled': model,
        'reference': model_reference
    }

    # validity and uniqueness
    for z_name, z_vec in Z.items():
        z_model = model_map[z_name]
        if z_name == 'labeled' or z_model is None:
            continue

        z_vec = torch.cat(z_vec, dim=0).to(device)
        z_mean, z_std = z_vec.mean(0), z_vec.std(0)

        validity, uniqueness = eval_validity_and_uniqueness(z_model, z_mean, z_std, nasbench, device=device)

        _metrics_list(metrics_res_dict[z_name], 'validity').append(validity)
        _metrics_list(metrics_res_dict[z_name], 'uniqueness').append(uniqueness)

        if verbose > 1:
            print('{}: Ratio of valid decodings from the prior: {:.4f}'.format(z_name, validity))
            print('{}: Ratio of unique decodings from the prior: {:.4f}'.format(z_name, uniqueness))

    # validation accuracy
    for m_name, m in model_map.items():
        if m is None:
            continue

        # labeled only eval
        if m_name == 'labeled':
            labeled_gen_2 = enumerate_validation_labeled(valid_labeled, labeled_batches=False)
            labeled_gen_full = enumerate_validation_labeled(valid_labeled, labeled_batches=True)

            val_loss = eval_labeled_validation(m, labeled_gen_full, device, config, loss_labeled)
            _metrics_list(metrics_res_dict[m_name], 'val_loss').append(val_loss)
            if verbose > 1:
                print(f"Validation labeled loss: {val_loss}")

            # evaluate reconstruction accuracy on labeled batches using unlabeled (original) model
            val_set = labeled_gen_2
            n_validation = len(valid_labeled)
            m = model_map['unlabeled']
        else:
            val_set = valid_unlabeled
            n_validation = len(valid_unlabeled)

        # common for all models
        if verbose > 1:
            print(f"Validation accuracy of the network - {m_name}:")
        eval_vae_validation(m, val_set, metrics_res_dict[m_name], device, config, verbose=verbose,
                            n_validation=n_validation)

    # TODO split this eval func

    for k, loss_dict in losses_total.items():
        epoch_means = mean_losses(losses_epoch[k])

        if verbose > 0:
            print('epoch {} loss: {} {}'.format(epoch, k, epoch_means))

        for loss_name, mean_val in epoch_means.items():
            loss_dict[loss_name].append(mean_val)


def checkpoint_metrics_losses(metrics, losses, save_dir, epoch):
    metrics_df = []
    for k, metrics in metrics.items():
        metrics_df.append({'model': k, **metrics})

    metrics_df = pd.DataFrame(metrics_df)
    metrics_df.to_csv(os.path.join(save_dir, f"metrics_epoch-{epoch}.csv"), index=False)

    loss_df = []
    for k, loss in losses.items():
        for loss_name, loss_values in loss.items():
            loss_df.append([f"{k}_{loss_name}", *loss_values])

    loss_df = pd.DataFrame(loss_df).T
    loss_df.columns = loss_df.iloc[0]
    loss_df.drop(index=0, inplace=True)
    loss_df.to_csv(os.path.join(save_dir, f"loss_epoch-{epoch}.csv"), index=False)


def mean_losses(loss_lists):
    return {k: np.mean(v) for k, v in loss_lists.items() if len(v)}


def _init_loss_lists():
    return {
        'total': [],
        'unlabeled': [],
        'labeled': []
    }


def init_stats_dict(kind=None):
    def get_value():
        if kind == 'loss':
            return _init_loss_lists()
        elif kind == 'metrics':
            return {}
        else:
            return []

    return {
        'labeled': get_value(),
        'unlabeled': get_value(),
        'reference': get_value()
    }
