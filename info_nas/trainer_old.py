# TODO napsat že source je arch2vec s úpravama víceméně
import os
import time

import numpy as np
import random
import torch
import torch.backends.cudnn
from info_nas.eval_old import mean_losses, eval_epoch, init_stats_dict, checkpoint_metrics_losses
from torch.utils.tensorboard import SummaryWriter

from arch2vec.models.model import VAEReconstructed_Loss
from info_nas.models.utils import save_extended_vae, get_optimizer
from torch import nn

from arch2vec.extensions.get_nasbench101_model import get_arch2vec_model
from arch2vec.utils import preprocessing, save_checkpoint_vae
from arch2vec.models.configs import configs

from info_nas.datasets.io.semi_dataset import get_train_valid_datasets
from info_nas.models.io_model import model_dict
from info_nas.config import local_model_cfg, load_json_cfg
from info_nas.metrics.losses import losses_dict


def train(labeled, unlabeled, nasbench, checkpoint_dir, transforms=None, valid_transforms=None,
          use_reference_model=False, model_config=None, device=None,
          batch_size=32, seed=1, epochs=8, writer=None, verbose=2, print_frequency=1000,
          batch_len_labeled=4, torch_deterministic=False, cudnn_deterministic=False):
    """
    Train the extended model on the labeled and unlabeled dataset. Optionally, train the original model alongside the
    extended one for reference. Save model checkpoints and metrics to a directory.

    Args:
        labeled: The labeled dataset (e.g from info_nas.datasets.arch2vec_dataset)
        unlabeled: The unlabeled dataset (e.g from info_nas.datasets.arch2vec_dataset)
        nasbench: An instance of nasbench.api.NASBench(nb_path).
        checkpoint_dir: The directory to save checkpoints in.
        transforms: Transforms for the train set.
        valid_transforms: Transforms for the valid set.
        use_reference_model: If True, train the reference model.
        model_config: Config for the training, if None, set to info_nas.configs.local_model_cfg
        device: Device for the training.
        batch_size: Batch size of both labeled and unlabeled batches.
        seed: Seed to use.
        epochs: Number of epochs
        writer: Not yet implemented SummaryWriter
        verbose: 0, 1 or 2, control the output
        print_frequency: How often to print the train loss.
        batch_len_labeled: Number of items in the labeled batch tuple
        torch_deterministic: Use deterministic torch
        cudnn_deterministic: Use deterministic cudnn

    Returns: Trained labeled model, metrics, loss stats

    """

    config, model_config = _init_config_and_seeds(model_config, seed, torch_deterministic, cudnn_deterministic)

    # TODO finish writer
    if writer is not None:
        writer = SummaryWriter(writer)

    # init dataset
    train_dataset, valid_labeled, valid_labeled_orig, valid_unlabeled = get_train_valid_datasets(
        labeled, unlabeled, batch_size=batch_size, labeled_transforms=transforms, val_batch_size=batch_size,
        labeled_val_transforms=valid_transforms, **model_config['dataset_config']
    )
    dataset_len = len(train_dataset)
    # precompute validation len
    n_valid_labeled_orig = 0
    for _ in valid_labeled_orig:
        n_valid_labeled_orig += 1

    # init models
    if not labeled['train']['use_reference']:
        in_channels = labeled['train']['inputs'].shape[1]
    else:
        in_channels = labeled['train']['dataset'].shape[1]

    # init models
    model, optimizer = get_arch2vec_model(device=device)
    model_labeled, optimizer_labeled = _initialize_labeled_model(model, in_channels, device=device,
                                                                 model_config=model_config)

    # train the reference model as well
    if use_reference_model:
        model_ref, optimizer_ref = get_arch2vec_model(device=device)
        model_ref.load_state_dict(model.state_dict())
    else:
        model_ref = None

    # init losses and logs
    loss_func_vae = VAEReconstructed_Loss(**config['loss'])
    loss_func_labeled = losses_dict[model_config['loss']](**model_config['loss_kwargs'])
    weight_vae = model_config['loss_vae_weight']  # VAE loss weight for labeled data (labeled loss unweighted)

    # stats for all three model variants (labeled, unlabeled, reference)
    loss_lists_total = init_stats_dict('loss')
    metrics_total = init_stats_dict('metrics')
    metrics_total['running_time'] = []
    start_time = time.process_time()

    for epoch in range(epochs):
        model.train()
        model_labeled.train()
        if use_reference_model:
            model_ref.train()

        n_labeled_batches, n_unlabeled_batches = 0, 0
        loss_lists_epoch = init_stats_dict('loss')
        Z = init_stats_dict()

        for i, batch in enumerate(train_dataset):
            # determine if labeled/unlabeled batch
            if len(batch) == 2:
                _train_on_batch(model, batch, optimizer, device, config, loss_func_vae, loss_func_labeled,
                                loss_lists_epoch['unlabeled'], Z['unlabeled'], eval_labeled=False)
                n_unlabeled_batches += 1

            elif len(batch) == batch_len_labeled:
                _train_on_batch(model_labeled, batch, optimizer_labeled, device, config, loss_func_vae,
                                loss_func_labeled, loss_lists_epoch['labeled'], Z['labeled'],
                                loss_vae_weight=weight_vae, eval_labeled=True)
                n_labeled_batches += 1

            else:
                raise ValueError(f"Invalid dataset - batch has {len(batch)} items, supported is 2 or "
                                 f"{batch_len_labeled}.")

            # train reference on unlabeled
            if use_reference_model:
                ref_weight = 1.0 if len(batch) == 2 else weight_vae  # reference model is trained on all batches
                _train_on_batch(model_ref, batch, optimizer_ref, device, config, loss_func_vae, loss_func_labeled,
                                loss_lists_epoch['reference'], Z['reference'],
                                loss_vae_weight=ref_weight, eval_labeled=False)

            # batch stats
            if verbose > 0 and i % print_frequency == 0:
                print(f'epoch {epoch}: batch {i} / {dataset_len}: ')
                for key, losses in loss_lists_epoch.items():
                    losses = ", ".join([f"{k}: {v}" for k, v in mean_losses(losses).items()])
                    print(f"\t {key}: {losses}")

                print(f'\t labeled batches: {n_labeled_batches}, unlabeled batches: {n_unlabeled_batches}')

        # epoch stats
        eval_epoch(model, model_labeled, model_ref, metrics_total, Z, loss_lists_total, loss_lists_epoch, epoch,
                   device, nasbench, valid_unlabeled, valid_labeled, valid_labeled_orig, config, model_config,
                   loss_func_labeled, verbose=verbose)

        metrics_total['running_time'].append(time.process_time() - start_time)

        checkpoint_metrics_losses(metrics_total, loss_lists_total, checkpoint_dir)

        # save network checkpoints
        make_checkpoint = 'checkpoint' in model_config and (epoch + 1) % model_config['checkpoint'] == 0
        if epoch == epochs - 1 or make_checkpoint:
            # save labeled/unlabeled models
            save_extended_vae(checkpoint_dir, model_labeled, optimizer_labeled, epoch,
                              model_config['model_class'], model_config['model_kwargs'])
            _save_arch2vec_model(model, optimizer, checkpoint_dir, 'orig', epoch)

            if use_reference_model:
                _save_arch2vec_model(model_ref, optimizer_ref, checkpoint_dir, 'ref', epoch)

        # TODO tensorboard?

    # TODO lepší zaznamenání výsledků
    return model_labeled, metrics_total, loss_lists_total


def _save_arch2vec_model(model, optimizer, checkpoint_dir, model_type, epoch):
    # keep the original function signature, save what I need
    orig_path = os.path.join(checkpoint_dir, f"model_{model_type}_epoch-{epoch}.pt")
    save_checkpoint_vae(model, optimizer, epoch, None, None, None, None, None, f_path=orig_path)


def _initialize_labeled_model(model, in_channels, model_config=None, device=None):
    model_class = model_dict[model_config['model_class']]

    model = model_class(model, in_channels, model_config['out_channels'], **model_config['model_kwargs'])
    if device is not None:
        model = model.to(device)

    optimizer = get_optimizer(model, **model_config['optimizer'])

    return model, optimizer


def _forward_batch(model, adj, ops, inputs=None):
    # forward
    if inputs is None:
        # unlabeled (original model)
        model_out = model(ops, adj.to(torch.long))
    else:
        # labeled (extended model)
        model_out = model(ops, adj.to(torch.long), inputs)

    return model_out


def _eval_batch(model_out, adj, ops, prep_reverse, loss, loss_labeled, loss_history, loss_vae_weight=1.0, outputs=None):
    ops_recon, adj_recon, mu, logvar = model_out[:4]

    adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
    adj, ops = prep_reverse(adj, ops)

    if outputs is not None:
        assert len(model_out) == 6  # TODO could differ
        outs_recon = model_out[-1]

        labeled_out = loss_labeled(outs_recon, outputs)
    else:
        labeled_out = None

    vae_out = loss((ops_recon, adj_recon), (ops, adj), mu, logvar)
    vae_out = loss_vae_weight * vae_out
    total_out = vae_out + labeled_out if labeled_out is not None else vae_out

    loss_history['total'].append(total_out.item())
    loss_history['unlabeled'].append(vae_out.item())
    if labeled_out is not None:
        loss_history['labeled'].append(labeled_out.item())

    return total_out


def _train_on_batch(model, batch, optimizer, device, config, loss_func_vae, loss_func_labeled, loss_list, Z,
                    loss_vae_weight=1.0, eval_labeled=False):

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
                           loss_list, loss_vae_weight=loss_vae_weight, outputs=outputs)

    loss_out.backward()

    nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()


def _init_config_and_seeds(model_config, seed, torch_deterministic, cudnn_deterministic):
    # io model config
    if model_config is None:
        model_config = local_model_cfg
    elif isinstance(model_config, str):
        model_config = load_json_cfg(model_config)

    # arch2vec config
    config = configs[model_config['arch2vec_config']]

    if torch_deterministic:
        torch.use_deterministic_algorithms(True)

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    return config, model_config
