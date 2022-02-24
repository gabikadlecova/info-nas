# TODO eval loss dict separate, eval models
import os

import numpy as np
import pandas as pd
import torch
from arch2vec.extensions.get_nasbench101_model import eval_validity_and_uniqueness, eval_validation_accuracy
from arch2vec.utils import preprocessing

from info_nas.models.losses import metrics_dict
from info_nas.models.utils import get_hash_accuracy


def _metrics_list(res_dict, key):
    return res_dict.setdefault(key, [])


def eval_vae_validation(mod, valid_set, res_dict, device, config, verbose=2):
    val_stats = eval_validation_accuracy(
        mod, valid_set, config=config, device=device,
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


def eval_labeled_validation(model, validation, device, config, loss_labeled, return_all_metrics=False, nasbench=None):
    if isinstance(validation, dict):
        if return_all_metrics:
            raise ValueError("Can return only summary metrics for multiple validation sets.")

        # process multiple validation sets
        res_dict = {}

        for val_name, val_set in validation.items():
            metrics = _eval_labeled_validation(model, val_set, device, config, loss_labeled, nasbench=nasbench)
            metrics = {f"{val_name}-{k}": v for k, v in metrics.items()}
            res_dict.update(metrics)

        return res_dict
    else:
        # there is only one validation set
        return _eval_labeled_validation(model, validation, device, config, loss_labeled,
                                        return_all_metrics=return_all_metrics, nasbench=nasbench)


def _eval_labeled_validation(model, validation, device, config, loss_labeled, return_all_metrics=False, nasbench=None):
    loss_m = {"val_loss": []}
    metrics = {k: [] for k in metrics_dict.keys()}
    metrics = {**loss_m, **metrics}

    print(f"Evaluating model on labeled validation set ({len(validation)} batches).")
    for batch in validation:

        if isinstance(batch, dict):
            adj, ops = batch['adj'], batch['ops']
            adj, ops = adj.to(device), ops.to(device)
            model_out = model(ops, adj)

            outputs = get_hash_accuracy(batch['hash'], nasbench, config)
        else:
            adj, ops, inputs, outputs = batch[:4]
            adj, ops = adj.to(device), ops.to(device)
            adj, ops, prep_reverse = preprocessing(adj, ops, **config['prep'])

            inputs, outputs = inputs.to(device), outputs.to(device)

            model_out = model(ops, adj.to(torch.long), inputs)

        assert len(model_out) == 6  # TODO could differ
        outs_recon = model_out[-1]

        labeled_out = loss_labeled(outs_recon, outputs)
        metrics["val_loss"].append(labeled_out.detach().item())

        for metric_k, metric in metrics_dict.items():
            metric_out = metric(outs_recon, outputs)
            metrics[metric_k].append(metric_out.detach().item())

    mean_metrics = {k: np.mean(m) for k, m in metrics.items()}
    mean_metrics["val_loss_min"] = np.min(metrics["val_loss"])
    mean_metrics["val_loss_max"] = np.max(metrics["val_loss"])
    mean_metrics["val_loss_std"] = np.std(metrics["val_loss"])
    mean_metrics["val_loss_median"] = np.median(metrics["val_loss"])
    return mean_metrics if not return_all_metrics else (mean_metrics, metrics)


def eval_epoch(model, model_labeled, model_reference, metrics_res_dict, Z, losses_total, losses_epoch, epoch, device,
               nasbench, valid_unlabeled, valid_labeled, valid_labeled_orig, config, loss_labeled, verbose=2):
    model.eval()
    model_labeled.eval()
    if model_reference is not None:
        model_reference.eval()

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
            val_metrics = eval_labeled_validation(m, valid_labeled, device, config, loss_labeled, nasbench=nasbench)
            for val_m_name, val_m_loss in val_metrics.items():
                _metrics_list(metrics_res_dict[m_name], val_m_name).append(val_m_loss)
                if verbose > 1:
                    print(f"Validation labeled - {val_m_name}: {val_m_loss}")

            # evaluate reconstruction accuracy on labeled batches using unlabeled (original) model
            val_set = valid_labeled_orig
            m = model_map['unlabeled']
        else:
            val_set = valid_unlabeled

        # common for all models
        if verbose > 1:
            print(f"Validation accuracy of the network - {m_name}:")
        eval_vae_validation(m, val_set, metrics_res_dict[m_name], device, config, verbose=verbose)

    # TODO split this eval func

    for k, loss_dict in losses_total.items():
        epoch_means = mean_losses(losses_epoch[k])

        if verbose > 0:
            print('epoch {} loss: {} {}'.format(epoch, k, epoch_means))

        for loss_name, mean_val in epoch_means.items():
            loss_dict[loss_name].append(mean_val)


def checkpoint_metrics_losses(metrics, losses, save_dir):
    metrics_df = []
    for k, metric_list in metrics.items():
        if k == "running_time":
            metrics_df.append(["running_time", *metric_list])
        else:
            for m_name, m_vals in metric_list.items():
                metrics_df.append([f"{k}_{m_name}", *m_vals])

    metrics_df = pd.DataFrame(metrics_df).T
    metrics_df.columns = metrics_df.iloc[0]
    metrics_df.drop(index=0, inplace=True)
    metrics_df.to_csv(os.path.join(save_dir, f"metrics.csv"))

    loss_df = []
    for k, loss in losses.items():
        for loss_name, loss_values in loss.items():
            loss_df.append([f"{k}_{loss_name}", *loss_values])

    loss_df = pd.DataFrame(loss_df).T
    loss_df.columns = loss_df.iloc[0]
    loss_df.drop(index=0, inplace=True)
    loss_df.to_csv(os.path.join(save_dir, f"loss.csv"))


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
