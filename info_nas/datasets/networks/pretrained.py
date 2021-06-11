from datetime import datetime
from typing import List

from nasbench_pytorch.model import Network as NBNetwork
from nasbench_pytorch.trainer import train, test
from info_nas.datasets.networks.utils import get_net_from_hash, save_trained_net, is_net_pretrained


def pretrain_network_dataset(net_hashes: List[str], nasbench, dataset, device=None, num_epochs=10, num_labels=10,
                             dir_path='./checkpoints/', skip_existing=True):

    train_set, n_train, val_set, n_val, test_set, n_test = dataset

    net_hashes = [n for n in net_hashes if not (skip_existing and is_net_pretrained(n, dir_path=dir_path))]
    print(f"Pretraining {len(net_hashes)} network{'s' if len(net_hashes) > 1 else ''}.\n----------------------------\n")

    for net_hash in net_hashes:
        print('--------------------')
        print(f"Pretraining network {net_hash}.")
        now = datetime.now()
        print(now.strftime("%d/%m/%Y %H:%M:%S\n--------------------"))

        ops, adjacency = get_net_from_hash(net_hash, nasbench)
        net = NBNetwork((adjacency, ops), num_labels)

        # train net and save it
        net = net.to(device)
        net, metrics = pretrain_network_cifar(net, train_set, val_set, test_set,
                                              num_tests=n_test, num_epochs=num_epochs, device=device)

        save_trained_net(net_hash, net, info=metrics, net_args=[num_labels], dir_path=dir_path)


def pretrain_network_cifar(net, train_loader, valid_loader, test_loader, num_tests=None, num_epochs=108, device=None):

    # TODO will be changed if more metrics needed
    loss, acc, val_loss, val_acc = train(net, train_loader, validation_loader=valid_loader, num_epochs=num_epochs,
                                         device=device)

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
