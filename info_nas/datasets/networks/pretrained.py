from nasbench_pytorch.trainer import train, test


# TODO nacitani natrenovanejch z directory


def pretrain_network_dataset(nets):
    # TODO vybrat nahodny site do poctu
    #  pak veci z jupyteru
    pass


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
