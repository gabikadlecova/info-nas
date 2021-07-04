# TODO napsat že source je arch2vec s úpravama víceméně
import numpy as np
import random
import torch
import torch.backends.cudnn
from torch.utils.tensorboard import SummaryWriter

from arch2vec.models.model import VAEReconstructed_Loss
from info_nas.models.utils import save_extended_vae
from torch import nn

from arch2vec.extensions.get_nasbench101_model import get_arch2vec_model
from arch2vec.extensions.get_nasbench101_model import eval_validity_and_uniqueness, eval_validation_accuracy
from arch2vec.utils import preprocessing
from arch2vec.models.configs import configs

from info_nas.datasets.io.semi_dataset import get_train_valid_datasets
from info_nas.models.io_model import model_dict
from info_nas.config import local_model_cfg, load_json_cfg
from info_nas.models.losses import losses_dict


def _initialize_labeled_model(model, in_channels, out_channels, model_config=None, device=None):
    model_class = model_dict[model_config['model_class']]

    model = model_class(model, in_channels, out_channels, **model_config['model_kwargs'])
    if device is not None:
        model = model.to(device)

    return model


def _forward_batch(model, adj, ops, inputs=None):
    # forward
    if inputs is None:
        # unlabeled (original model)
        model_out = model(ops, adj.to(torch.long))
    else:
        # labeled (extended model)
        model_out = model(ops, adj.to(torch.long), inputs)

    return model_out


def _eval_batch(model_out, adj, ops, prep_reverse, loss, loss_labeled, loss_history, outputs=None):
    ops_recon, adj_recon, mu, logvar, = model_out[:4]

    adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
    adj, ops = prep_reverse(adj, ops)

    if outputs is not None:
        assert len(model_out) == 6
        outs_recon = model_out[-1]

        labeled_out = loss_labeled(outs_recon, outputs)
    else:
        labeled_out = None

    vae_out = loss((ops_recon, adj_recon), (ops, adj), mu, logvar)
    total_out = vae_out + labeled_out if labeled_out is not None else vae_out

    loss_history['total'].append(total_out.item())
    loss_history['unlabeled'].append(vae_out.item())
    if labeled_out is not None:
        loss_history['labeled'].append(labeled_out.item())

    return total_out


# TODO eval loss dict separate, eval models

def _eval_epoch(model, model_labeled, model_reference, metrics_res_dict, Z, losses_total, losses_epoch, epoch, device,
                nasbench, valid_unlabeled, config, verbose=2):
    def metrics_list(res_dict, key):
        return res_dict.setdefault(key, [])

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

        metrics_list(metrics_res_dict[z_name], 'validity').append(validity)
        metrics_list(metrics_res_dict[z_name], 'uniqueness').append(uniqueness)

        if verbose > 1:
            print('{}: Ratio of valid decodings from the prior: {:.4f}'.format(z_name, validity))
            print('{}: Ratio of unique decodings from the prior: {:.4f}'.format(z_name, uniqueness))

    # validation accuracy
    for m_name, m in model_map.items():
        if m is None:
            continue

        if m_name == 'labeled':
            continue # TODO!!!
        else:
            val_stats = eval_validation_accuracy(
                m, valid_unlabeled, config=config, device=device
            )

            stats_names = ['acc_ops_val', 'mean_corr_adj_val', 'mean_fal_pos_adj_val', 'acc_adj_val']
            for n, val in zip(stats_names, val_stats):
                metrics_list(metrics_res_dict[m_name], n).append(val)

            if verbose > 1:
                print(
                    'validation set: acc_ops:{0:.4f}, mean_corr_adj:{1:.4f}, mean_fal_pos_adj:{2:.4f}, acc_adj:{3:.4f}'.format(
                        *val_stats
                    )
                )

    # TODO loss to metrics?
    # TODO res to pandas
    # TODO split this eval func

    for k, loss_dict in losses_total.items():
        epoch_means = _mean_losses(losses_epoch[k])

        if verbose > 0:
            print('epoch {} loss: {} {}'.format(epoch, k, epoch_means))

        for loss_name, mean_val in epoch_means.items():
            loss_dict[loss_name].append(mean_val)


def _mean_losses(loss_lists):
    return {k: np.mean(v) for k, v in loss_lists.items() if len(v)}


# TODO remove labeled hashes from unlabeled


def _train_on_batch(model, batch, optimizer, device, config, loss_func_vae, loss_func_labeled, loss_list, Z,
                    eval_labeled=False):

    optimizer.zero_grad()

    # adj, ops preprocessing
    adj, ops = batch[0], batch[1]
    adj, ops = adj.to(device), ops.to(device)
    adj, ops, prep_reverse = preprocessing(adj, ops, **config['prep'])

    # labeled vs unlabeled batches
    if eval_labeled:
        inputs, outputs = batch[2].to(device), batch[3].to(device)
    else:
        inputs, outputs = None, None

    # forward
    model_out = _forward_batch(model, adj, ops, inputs=inputs)
    mu = model_out[2]
    Z.append(mu.cpu())

    loss_out = _eval_batch(model_out, adj, ops, prep_reverse, loss_func_vae, loss_func_labeled,
                           loss_list, outputs=outputs)

    loss_out.backward()

    nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()


def _init_config_and_seeds(config, model_config, seed, torch_deterministic, cudnn_deterministic):
    # arch2vec config
    config = configs[config]

    # io model config
    if model_config is None:
        model_config = local_model_cfg
    elif isinstance(model_config, str):
        model_config = load_json_cfg(model_config)

    if torch_deterministic:
        torch.use_deterministic_algorithms(True)

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    return config, model_config


def train(labeled, unlabeled, nasbench, checkpoint_path, use_reference_model=False, model_config=None, device=None,
          batch_size=32, k=1, n_workers=0, n_val_workers=0, seed=1, epochs=8, config=4, print_frequency=1000,
          torch_deterministic=False, cudnn_deterministic=False, writer=None, verbose=2):

    config, model_config = _init_config_and_seeds(config, model_config, seed, torch_deterministic, cudnn_deterministic)

    # TODO finish writer
    if writer is not None:
        writer = SummaryWriter(writer)

    # init dataset
    train_dataset, valid_labeled, valid_unlabeled = get_train_valid_datasets(labeled, unlabeled, k=k,
                                                                             batch_size=batch_size, n_workers=n_workers,
                                                                             n_valid_workers=n_val_workers)
    dataset_len = len(train_dataset)

    # init models
    if not labeled['train_io']['use_reference']:
        in_channels = labeled['train_io']['inputs'].shape[1]
    else:
        in_channels = labeled['train_io']['dataset'].shape[1]

    out_channels = labeled['train_io']['outputs'].shape[1]

    model, optimizer = get_arch2vec_model(device=device)
    model_labeled = _initialize_labeled_model(model, in_channels, out_channels,
                                              device=device, model_config=model_config)
    if use_reference_model:
        model_ref, optimizer_ref = get_arch2vec_model(device=device)
        model_ref.load_state_dict(model.state_dict())
    else:
        model_ref = None

    # init losses and logs
    loss_func_vae = VAEReconstructed_Loss(**config['loss'])
    loss_func_labeled = losses_dict[model_config['loss']]

    # stats for all three model variants
    def init_loss_lists():
        return {
            'total': [],
            'unlabeled': [],
            'labeled': []
        }

    def init_stats_dict(use_loss_list=True):
        return {
            'labeled': init_loss_lists() if use_loss_list else [],
            'unlabeled': init_loss_lists() if use_loss_list else [],
            'reference': init_loss_lists() if use_loss_list else []
        }

    loss_lists_total = init_stats_dict()
    metrics_total = {
        'labeled': {},
        'unlabeled': {},
        'reference': {}
    }

    for epoch in range(epochs):
        model.train()
        model_labeled.train()

        n_labeled_batches, n_unlabeled_batches = 0, 0

        #TODO metrics dict
        #TODO

        loss_lists_epoch = init_stats_dict()
        Z = init_stats_dict(use_loss_list=False)

        for i, batch in enumerate(train_dataset):
            # determine if labeled/unlabeled batch
            if len(batch) == 2:
                extended_model = model
                loss_list = loss_lists_epoch['unlabeled']
                Z_list = Z['unlabeled']
                is_labeled = False

                n_unlabeled_batches += 1

            elif len(batch) == 4:
                extended_model = model_labeled
                loss_list = loss_lists_epoch['labeled']
                Z_list = Z['labeled']
                is_labeled = True

                n_labeled_batches += 1
            else:
                raise ValueError(f"Invalid dataset - batch has {len(batch)} items, supported is 2 or 4.")

            # train models
            _train_on_batch(extended_model, batch, optimizer, device, config, loss_func_vae, loss_func_labeled,
                            loss_list, Z_list, eval_labeled=is_labeled)
            if use_reference_model:
                _train_on_batch(model_ref, batch, optimizer_ref, device, config, loss_func_vae, loss_func_labeled,
                                loss_lists_epoch['reference'], Z['reference'], eval_labeled=False)

            # batch stats
            if verbose > 0 and i % print_frequency == 0:
                print(f'epoch {epoch}: batch {i} / {dataset_len}: ')
                for key, losses in loss_lists_epoch.items():
                    losses = ", ".join([f"{k}: {v}" for k, v in _mean_losses(losses).items()])
                    print(f"\t {key}: {losses}")

                print(f'\t labeled batches: {n_labeled_batches}, unlabeled batches: {n_unlabeled_batches}')

        # epoch stats
        make_checkpoint = 'checkpoint' in model_config and epoch % model_config['checkpoint'] == 0
        if epoch == epochs + 1 or make_checkpoint:
            save_extended_vae(checkpoint_path, model_labeled, optimizer, epoch,
                              model_config['model_class'], model_config['model_kwargs'])

            # TODO checkpoint metrics

        _eval_epoch(model, model_labeled, model_ref, metrics_total, Z, loss_lists_total, loss_lists_epoch, epoch,
                    device, nasbench, valid_unlabeled, config, verbose=verbose)

        # TODO tensorboard?

    # TODO lepší zaznamenání výsledků
    return model_labeled, metrics_total


# TODO pretrain model MOJE:
#  - load nasbench, get nb dataset, get MY io dataset (x)
#  - get model and optimizer (x) ; get MY model (x) and MY loss (x)
#  - for epoch in range(epochs): (x)
#      - for batch in batches: (x)
#         - ops, adj to cuda, PREPRO (x)
#         - forward and backward (x)
#         - (take care of my loss) (x)
#      - validity, uniqueness, val_accuracy (x)
#      - LOSS TOTAL, CHECKPOINT (x) (x)


