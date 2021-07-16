import json

local_dataset_cfg = {
    'cifar-10': {
        'batch_size': 32,
        'validation_size': 1000,
        'num_workers': 8
    },

    'nb_dataset': {
        'test_size': 0.1
    },

    'pretrain': {
        'num_epochs': 10
    },
    'io': {
        'nth_input': 0,
        'nth_output': -2,
        'loss': None
    }
}


local_model_cfg = {
    'model_class': 'concat',
    'model_kwargs': {
        'n_steps': 2,
        'n_convs': 2,
        'activation': 'linear'
    },
    'optimizer': {
        'name': 'Adam',
        'lr': 1e-4,
        'betas': (0.9, 0.999),
        'eps': 1e-08
    },
    'out_channels': 513,
    'loss': 'MSE',
    'loss_kwargs': {},
    'loss_vae_weight': 1.0,
    'checkpoint': 5,
    'dataset_config': {
        'k': 20,
        'repeat_unlabeled': 1,
        'n_workers': 4,
        'n_valid_workers': 4
    },
    'arch2vec_config': 4
}

# TODO extend the config with missing fields (default settings)


def load_json_cfg(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)
