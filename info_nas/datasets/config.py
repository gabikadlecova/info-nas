cfg = {
    'cifar-10': {
        'batch_size': 32,
        'validation_size': 1000,
        'num_workers': 8
    },

    'nb_dataset': {
        'test_size': 0.1,
        'batch_size': 32,
        'val_batch_size': 100
    },

    'pretrain': {
        'num_epochs': 10,
        'num_labels': 10
    },
    'io': {
        'nth_input': 0,
        'nth_output': -2,
        'loss': None
    }
}
