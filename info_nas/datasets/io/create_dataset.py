from nasbench_pytorch.trainer import train, test
from nasbench_pytorch.datasets.cifar10 import prepare_dataset


def create_dataset(dataset, networks):

    # TODO train valid test tady (ať je jen jednou)

    # vymyslet kam dat ty predtrenovany
    pass


def pretrain_network_cifar(net, train_loader, valid_loader, test_loader, num_tests=None):

    train(net, train_loader, valid_loader)
    test(net, test_loader, num_tests=num_tests)

    # TODO ještě ať to vrací train a test loss apod (kuk nasbench, co je přesně train loss? last epoch?)
