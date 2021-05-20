from nasbench_pytorch.trainer import train, test


# TODO tady bude taky nejaky get_nth_output
# TODO nacitani


def pretrain_network_cifar(net, train_loader, valid_loader, test_loader, num_tests=None):

    loss, acc, val_loss, val_acc = train(net, train_loader, valid_loader)  # TODO will be changed if more metrics needed
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