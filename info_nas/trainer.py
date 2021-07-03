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

    loss_history.append({
        'total': total_out.item(),
        'unlabeled': vae_out.item(),
        'labeled': labeled_out.item()
    })

    return total_out

"""
loss_epoch.append(loss.item())
            if verbose > 0 and i % print_frequency == 0:
                print('epoch {}: batch {} / {}: loss: {:.5f}, loss_labeled: {:.5f}'.format(epoch, i,
                                                                                           dataset_len,
                                                                                           loss_epoch[-1],
                                                                                           loss_epoch_labeled[-1]))

                print(f'epoch {epoch}: labeled batches: {n_labeled_batches}, unlabeled batches: {n_unlabeled_batches}')

        Z = torch.cat(Z, dim=0)
        z_mean, z_std = Z.mean(0), Z.std(0)

        validity, uniqueness = eval_validity_and_uniqueness(model, z_mean, z_std, nasbench, device=device)

        if verbose > 1:
            print('Ratio of valid decodings from the prior: {:.4f}'.format(validity))
            print('Ratio of unique decodings from the prior: {:.4f}'.format(uniqueness))

        # TODO validation set for LABELED
        acc_ops_val, mean_corr_adj_val, mean_fal_pos_adj_val, acc_adj_val = eval_validation_accuracy(model,
                                                                                                     valid_unlabeled,
                                                                                                     config=config,
                                                                                                     device=device)

        if verbose > 1:
            print(
                'validation set: acc_ops:{0:.4f}, mean_corr_adj:{1:.4f}, mean_fal_pos_adj:{2:.4f}, acc_adj:{3:.4f}'.format(
                    acc_ops_val, mean_corr_adj_val, mean_fal_pos_adj_val, acc_adj_val
                )
            )

        if verbose > 0:
            print('epoch {}: average loss {:.5f}'.format(epoch, sum(loss_epoch) / len(loss_epoch)))

        loss_total.append(sum(loss_epoch) / len(loss_epoch))
        loss_total_labeled.append(sum(loss_epoch_labeled) / len(loss_epoch_labeled))
"""

# TODO remove labeled hashes from unlabeled

# TODO zkusit trénovat paralelně model s io i bez io?


def _train_on_batch(model, batch, optimizer, device, config, loss_func_vae, loss_func_labeled, loss_list,
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
    in_channels, out_channels = labeled['train_io']['inputs'].shape[1], labeled['train_io']['outputs'].shape[1]

    model, optimizer = get_arch2vec_model(device=device)
    model_labeled = _initialize_labeled_model(model, in_channels, out_channels,
                                              device=device, model_config=model_config)
    if use_reference_model:
        model_ref, optimizer_ref = get_arch2vec_model(device=device)
        model_ref.load_state_dict(model.state_dict())

    # init losses and logs
    loss_func_vae = VAEReconstructed_Loss(**config['loss'])
    loss_func_labeled = losses_dict[model_config['loss']]

    loss_lists_total = {
        'labeled': [],
        'unlabeled': [],
        'reference': []
    }

    for epoch in range(epochs):
        model.train()
        model_labeled.train()

        n_labeled_batches, n_unlabeled_batches = 0, 0

        #TODO metrics dict
        #TODO

        loss_lists_epoch = {
            'labeled': [],
            'unlabeled': [],
            'reference': []
        }

        Z = []

        for i, batch in enumerate(train_dataset):
            if len(batch) == 2:
                extended_model = model
                loss_list = loss_lists_epoch['unlabeled']
                is_labeled = False

                n_unlabeled_batches += 1

            elif len(batch) == 4:
                extended_model = model_labeled
                loss_list = loss_lists_epoch['labeled']
                is_labeled = True

                n_labeled_batches += 1
            else:
                raise ValueError(f"Invalid dataset - batch has {len(batch)} items, supported is 2 or 4.")

            _train_on_batch(extended_model, batch, optimizer, device, config, loss_func_vae, loss_func_labeled,
                            loss_list, eval_labeled=is_labeled)

            if use_reference_model:
                _train_on_batch(model_ref, batch, optimizer_ref, device, config, loss_func_vae, loss_func_labeled,
                                loss_lists_epoch['reference'], eval_labeled=False)

        make_checkpoint = 'checkpoint' in model_config and epoch % model_config['checkpoint'] == 0
        if epoch == epochs + 1 or make_checkpoint:
            save_extended_vae(checkpoint_path, model_labeled, optimizer, epoch, loss_total[-1], loss_total_labeled[-1],
                              model_config['model_class'], model_config['model_kwargs'])

        # TODO eval epoch + eval reference possibly (all losses in loss dicts, then move it to total. also prints.).

        # TODO tensorboard?


    if verbose > 0:
        print('loss for epochs: \n', loss_total)
        print('labeled loss for epochs: \n', loss_total_labeled)
    # TODO lepší zaznamenání výsledků

    # TODO return more things
    return model_labeled


# TODO pretrain model MOJE:
#  - load nasbench, get nb dataset, get MY io dataset (x)
#  - get model and optimizer (x) ; get MY model (x) and MY loss
#  - for epoch in range(epochs): (x)
#      - for batch in batches: (x)
#         - ops, adj to cuda, PREPRO (x)
#         - forward and backward (x)
#         - (take care of my loss) (x)
#      - validity, uniqueness, val_accuracy (x)
#      - LOSS TOTAL, CHECKPOINT


