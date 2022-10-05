import time

import torch
import torch.nn as nn

from arch2vec.models.model import VAEReconstructed_Loss
from arch2vec.extensions.get_nasbench101_model import get_arch2vec_model
from arch2vec.utils import preprocessing, save_checkpoint_vae

from info_nas.datasets.io.semi_dataset import get_train_valid_datasets
from info_nas.eval_old import init_stats_dict, mean_losses, checkpoint_metrics_losses, eval_epoch
from info_nas.models.layers import LatentNodesFlatten, get_dense_list
from info_nas.models.utils import get_optimizer, save_extended_vae, get_hash_accuracy
from info_nas.trainer_old import _init_config_and_seeds, _save_arch2vec_model, _eval_batch


class AccuracyModel(nn.Module):
    def __init__(self, vae_model, is_log_accuracy=False, z_hidden=16, n_dense=1, n_hidden=512, dropout=None):
        super().__init__()
        self.vae_model = vae_model
        self.process_z = LatentNodesFlatten(self.vae_model.latent_dim, z_hidden=z_hidden)

        self.first_dense = nn.Linear(z_hidden, n_hidden)
        self.dense_list = get_dense_list(n_dense, dropout, n_hidden, 1)

        self.activation = None if is_log_accuracy else nn.Sigmoid()

    def predict_accuracy(self, z):
        z = self.process_z(z)
        z = self.first_dense(z)
        z = self.dense_list(z)

        if self.activation is not None:
            z = self.activation(z)

        return z.flatten()

    def forward(self, ops, args):
        ops_recon, adj_recon, mu, logvar, z = self.vae_model.forward(ops, args)
        accuracy = self.predict_accuracy(z)

        return ops_recon, adj_recon, mu, logvar, z, accuracy


def _train_on_batch(model, batch, optimizer, device, config, Z, loss_func_vae, loss_func_labeled, loss_list,
                    loss_vae_weight=1.0, accuracy=None):
    optimizer.zero_grad()

    # adj, ops preprocessing
    adj, ops = batch[0], batch[1]
    adj, ops = adj.to(device), ops.to(device)
    adj, ops, prep_reverse = preprocessing(adj, ops, **config['prep'])

    # forward
    model_out = model(ops, adj.to(torch.long))
    mu = model_out[2]
    Z.append(mu.cpu())

    loss_out = _eval_batch(model_out, adj, ops, prep_reverse, loss_func_vae, loss_func_labeled,
                           loss_list, loss_vae_weight=loss_vae_weight, outputs=accuracy)

    loss_out.backward()

    nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()


def train_as_infonas(labeled, unlabeled, nasbench, checkpoint_dir, transforms=None, valid_transforms=None,
              model_config=None, device=None, batch_size=32, seed=1, epochs=8, verbose=2, print_frequency=1000,
              torch_deterministic=False, cudnn_deterministic=False, is_log_accuracy=False):

        config, model_config = _init_config_and_seeds(model_config, seed, torch_deterministic, cudnn_deterministic)

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
        model, optimizer = get_arch2vec_model(device=device)
        model_labeled = AccuracyModel(model, is_log_accuracy=is_log_accuracy)
        model_labeled = model_labeled.to(device)
        optimizer_labeled = get_optimizer(model_labeled, **model_config['optimizer'])

        # init losses and logs
        loss_func_vae = VAEReconstructed_Loss(**config['loss'])
        loss_func_labeled = nn.MSELoss()
        weight_vae = model_config['loss_vae_weight']

        # stats for all three model variants (labeled, unlabeled, reference)
        loss_lists_total = init_stats_dict('loss')
        metrics_total = init_stats_dict('metrics')
        metrics_total['running_time'] = []
        start_time = time.process_time()

        for epoch in range(epochs):
            model.train()
            model_labeled.train()

            n_labeled_batches, n_unlabeled_batches = 0, 0
            loss_lists_epoch = init_stats_dict('loss')
            Z = init_stats_dict()

            for i, batch in enumerate(train_dataset):
                if isinstance(batch, dict):
                    net_hash = batch['hash']
                    batch = batch['adj'], batch['ops']
                    batch_acc = get_hash_accuracy(net_hash, nasbench, model_config, device=device)

                    _train_on_batch(model_labeled, batch, optimizer_labeled, device, config, Z['labeled'],
                                    loss_func_vae, loss_func_labeled, loss_lists_epoch['labeled'],
                                    loss_vae_weight=weight_vae, accuracy=batch_acc)
                    n_labeled_batches += 1
                else:
                    _train_on_batch(model, batch, optimizer, device, config, Z['unlabeled'], loss_func_vae,
                                    loss_func_labeled, loss_lists_epoch['unlabeled'])
                    n_unlabeled_batches += 1

                # batch stats
                if verbose > 0 and i % print_frequency == 0:
                    print(f'epoch {epoch}: batch {i} / {dataset_len}: ')
                    for key, losses in loss_lists_epoch.items():
                        losses = ", ".join([f"{k}: {v}" for k, v in mean_losses(losses).items()])
                        print(f"\t {key}: {losses}")

                    print(f'\t labeled batches: {n_labeled_batches}, unlabeled batches: {n_unlabeled_batches}')

            # epoch stats
            eval_epoch(model, model_labeled, None, metrics_total, Z, loss_lists_total, loss_lists_epoch, epoch,
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

        return model_labeled, metrics_total, loss_lists_total
