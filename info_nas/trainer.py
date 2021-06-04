# TODO napsat že source je arch2vec s úpravama víceméně
import torch
from torch import nn

from arch2vec.extensions.get_nasbench101_model import get_arch2vec_model, get_nasbench_datasets
from arch2vec.extensions.get_nasbench101_model import eval_validity_and_uniqueness, eval_validation_accuracy
from arch2vec.utils import preprocessing
from arch2vec.models.configs import configs


def _initialize_arch2vec_model(model):
    # TODO tady arch2vec + můj model
    return model


# TODO zkusit trénovat paralelně model s io i bez io?

def train(nb_dataset, nasbench, device=None, seed=1, val_size=0.1, epochs=8, batch_size=32, val_batch_size=100,
          config=4):
    # nb_dataset is path to json or dict

    config = configs[config]

    model, optimizer = get_arch2vec_model(device=device)
    model = _initialize_arch2vec_model(model)
    # TODO io dataset, io model

    nb_dataset = get_nasbench_datasets(nb_dataset, seed=seed, batch_size=batch_size, val_batch_size=val_batch_size,
                                       test_size=val_size)

    hash_train, X_adj_train, X_ops_train, indices_train = nb_dataset["train"]
    hash_val, X_adj_val, X_ops_val, indices_val = nb_dataset["val"]
    n_train, n_val = nb_dataset["n_train"], nb_dataset["n_val"]

    loss_total = []
    for epoch in range(epochs):
        model.train()

        # TODO jinej dataset
        # TODO shuffle (func kwarg)
        loss_epoch = []
        Z = []
        for i, (adj, ops, ind) in enumerate(zip(X_adj_train, X_ops_train, indices_train)):
            optimizer.zero_grad()
            adj, ops = adj.to(device), ops.to(device)

            # preprocessing
            adj, ops, prep_reverse = preprocessing(adj, ops, **config['prep'])

            # forward
            ops_recon, adj_recon, mu, logvar = model(ops, adj.to(torch.long))
            Z.append(mu)

            adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
            adj, ops = prep_reverse(adj, ops)

            # TODO loss
            #loss = VAEReconstructed_Loss(**cfg['loss'])((ops_recon, adj_recon), (ops, adj), mu, logvar)
            #loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            # TODO prints
            #loss_epoch.append(loss.item())
            #if i % 1000 == 0:
            #    print('epoch {}: batch {} / {}: loss: {:.5f}'.format(epoch, i, chunks, loss.item()))

        Z = torch.cat(Z, dim=0)
        z_mean, z_std = Z.mean(0), Z.std(0)

        validity, uniqueness = eval_validity_and_uniqueness(model, z_mean, z_std, nasbench, device=device)

        # TODO prints
        print('Ratio of valid decodings from the prior: {:.4f}'.format(validity))
        print('Ratio of unique decodings from the prior: {:.4f}'.format(uniqueness))

        acc_ops_val, mean_corr_adj_val, mean_fal_pos_adj_val, acc_adj_val = eval_validation_accuracy(model,
                                                                                                     X_adj_val,
                                                                                                     X_ops_val,
                                                                                                     indices_val,
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
#  - load nasbench, get nb dataset, get MY io dataset
#  - get model and optimizer; get MY model and MY loss
#  - for epoch in range(epochs):
#      - for batch in batches:
#         - ops, adj to cuda, PREPRO
#         - forward and backward (take care of my loss)
#      - validity, uniqueness, val_accuracy
#      - LOSS TOTAL, CHECKPOINT
#
# TODO ... ještě jednou zkontrolovat dle pretrain_nasbench_101

