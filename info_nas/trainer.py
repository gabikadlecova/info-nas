# TODO napsat že source je arch2vec s úpravama víceméně
import numpy as np
import random
import torch
import torch.backends.cudnn

from arch2vec.models.model import VAEReconstructed_Loss
from torch import nn

from arch2vec.extensions.get_nasbench101_model import get_arch2vec_model
from arch2vec.extensions.get_nasbench101_model import eval_validity_and_uniqueness, eval_validation_accuracy
from arch2vec.utils import preprocessing
from arch2vec.models.configs import configs

from info_nas.datasets.io.semi_dataset import get_train_valid_datasets
from info_nas.models.conv_embeddings import SimpleConvModel


def _initialize_labeled_model(model, in_channels, out_channels, **kwargs):
    # TODO config for kwargs
    return SimpleConvModel(model, in_channels, out_channels, **kwargs)


# TODO zkusit trénovat paralelně model s io i bez io?

def train(labeled, unlabeled, nasbench, device=None, batch_size=32, k=1, n_workers=0, n_val_workers=0, seed=1,
          epochs=8, config=4, print_frequency=1000, torch_deterministic=False, cudnn_deterministic=False):

    config = configs[config]

    if torch_deterministic:
        torch.use_deterministic_algorithms(True)

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    in_channels, out_channels = labeled['train_io']['inputs'].shape[1], labeled['train_io']['outputs'].shape[1]

    train_dataset, valid_labeled, valid_unlabeled = get_train_valid_datasets(labeled, unlabeled, k=k,
                                                                             batch_size=batch_size, n_workers=n_workers,
                                                                             n_valid_workers=n_val_workers)

    model, optimizer = get_arch2vec_model(device=device)
    model_labeled = _initialize_labeled_model(model, in_channels, out_channels)  # TODO config and kwargs

    dataset_len = len(train_dataset)
    loss_total = []
    for epoch in range(epochs):
        model.train()

        loss_epoch = []
        Z = []
        for i, batch in enumerate(train_dataset):
            if len(batch) == 2:
                adj, ops = batch
            elif len(batch) == 4:
                adj, ops, inputs, outputs = batch
                inputs, outputs = inputs.to(device), outputs.to(device)
            else:
                raise ValueError(f"Invalid dataset - batch has {len(batch)} items, supported is 2 or 4.")

            optimizer.zero_grad()

            # preprocessing
            adj, ops = adj.to(device), ops.to(device)
            adj, ops, prep_reverse = preprocessing(adj, ops, **config['prep'])

            # forward
            if len(batch) == 2:
                # unlabeled (original model)
                ops_recon, adj_recon, mu, logvar = model(ops, adj.to(torch.long))
                Z.append(mu)

                loss = None
            else:
                # labeled (extended model)
                ops_recon, adj_recon, mu, logvar, outs_recon = model_labeled(ops, adj.to(torch.long), inputs)

                # TODO Z by byly dost biased, leda mít zvlášť Z_labeled a převážit to
                # Z.append(mu)

                # TODO loss!!
                loss = None

            adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
            adj, ops = prep_reverse(adj, ops)

            vae_loss = VAEReconstructed_Loss(**config['loss'])((ops_recon, adj_recon), (ops, adj), mu, logvar)
            loss = vae_loss if loss is None else vae_loss + loss
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            loss_epoch.append(loss.item())
            if i % print_frequency == 0:
                print('epoch {}: batch {} / {}: loss: {:.5f}'.format(epoch, i, dataset_len, loss_epoch[-1]))

        Z = torch.cat(Z, dim=0)
        z_mean, z_std = Z.mean(0), Z.std(0)

        validity, uniqueness = eval_validity_and_uniqueness(model, z_mean, z_std, nasbench, device=device)

        # TODO prints
        print('Ratio of valid decodings from the prior: {:.4f}'.format(validity))
        print('Ratio of unique decodings from the prior: {:.4f}'.format(uniqueness))

        # TODO validation set, modify eval function
        acc_ops_val, mean_corr_adj_val, mean_fal_pos_adj_val, acc_adj_val = eval_validation_accuracy(model,
                                                                                                     X_adj_val,
                                                                                                     X_ops_val,
                                                                                                     n_val,
                                                                                                     config=config,
                                                                                                     device=device)

        # TODO print
        print(
            'validation set: acc_ops:{0:.4f}, mean_corr_adj:{1:.4f}, mean_fal_pos_adj:{2:.4f}, acc_adj:{3:.4f}'.format(
                acc_ops_val, mean_corr_adj_val, mean_fal_pos_adj_val, acc_adj_val
            )
        )
        print('epoch {}: average loss {:.5f}'.format(epoch, sum(loss_epoch) / len(loss_epoch)))

        loss_total.append(sum(loss_epoch) / len(loss_epoch))
        # TODO checkpoint, jen jednou za x

        # TODO tensorboard?

    print('loss for epochs: \n', loss_total)
    # TODO ještě lepší zaznamenání výsledků


# TODO pretrain model MOJE:
#  - load nasbench, get nb dataset, get MY io dataset (x)
#  - get model and optimizer (x) ; get MY model (x) and MY loss
#  - for epoch in range(epochs): (x)
#      - for batch in batches: (x)
#         - ops, adj to cuda, PREPRO (x)
#         - forward and backward (x)
#         - (take care of my loss)
#      - validity, uniqueness, val_accuracy
#      - LOSS TOTAL, CHECKPOINT
#
# TODO ... ještě jednou zkontrolovat dle pretrain_nasbench_101

