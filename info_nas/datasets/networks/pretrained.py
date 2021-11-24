from datetime import datetime
from typing import List

from nasbench_pytorch.model import Network as NBNetwork
from nasbench_pytorch.trainer import train, test
from info_nas.datasets.networks.utils import get_net_from_hash, save_trained_net, is_net_pretrained


def pretrain_network_dataset(net_hashes: List[str], nasbench, dataset, device=None, num_epochs=10, num_labels=10,
                             dir_path='./checkpoints/', skip_existing=True, **kwargs):
    """
    Pretrain networks from the NAS-Bench-101 dataset according to a list of their hashes. Store their checkpoints.

    Args:
        net_hashes: A list of net hashes that should be trained.
        nasbench: Path to the saved arch2vec dataset (will be created if it does not exist).
        dataset: Dataset to use for the training (see info_nas.datasets.io.dataset_from_pretrained() for the
            format).

        device: Device to use for the training.
        num_epochs: Number of training epochs.
        num_labels: Number of labels for the classification.
        dir_path: Path where the checkpoints will be saved to.
        skip_existing: Skip networks that exists in the directory.
        **kwargs: Additional kwargs for the training.

    """

    train_set, n_train, val_set, n_val, test_set, n_test = dataset

    net_hashes = [n for n in net_hashes if not (skip_existing and is_net_pretrained(n, dir_path=dir_path))]
    print(f"Pretraining {len(net_hashes)} network{'s' if len(net_hashes) > 1 else ''}.\n----------------------------\n")

    for i, net_hash in enumerate(net_hashes):
        print('--------------------')
        print(f"Pretraining network {i + 1}/{len(net_hashes)}: {net_hash}.")
        now = datetime.now()
        print(now.strftime("%d/%m/%Y %H:%M:%S\n--------------------"))

        # function for periodic checkpointing
        def periodic_checkpoint(network, metric_dict):
            save_trained_net(net_hash, network, info=metric_dict, net_args=[num_labels], dir_path=dir_path)

        ops, adjacency = get_net_from_hash(net_hash, nasbench)
        net = NBNetwork((adjacency, ops), num_labels)

        # train net and save it
        net = net.to(device)
        net, metrics = pretrain_network_cifar(net, train_set, val_set, test_set, n_labels=num_labels,
                                              num_tests=n_test, num_epochs=num_epochs, device=device, **kwargs)

        periodic_checkpoint(net, metrics)


def pretrain_network_cifar(net, train_loader, valid_loader, test_loader, num_tests=None, num_epochs=108, device=None,
                           print_frequency=50, save_every_k=None, checkpoint_func=None, **kwargs):
    # TODO will be changed if more metrics needed
    loss, acc, val_loss, val_acc = train(net, train_loader, validation_loader=valid_loader, num_epochs=num_epochs,
                                         device=device, print_frequency=print_frequency,
                                         checkpoint_every_k=save_every_k, checkpoint_func=checkpoint_func,
                                         **kwargs)

    test_loss, test_acc = test(net, test_loader, num_tests=num_tests)

    metrics = {
        'train_loss': loss,
        'train_accuracy': acc,
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
    }

    return net, metrics
