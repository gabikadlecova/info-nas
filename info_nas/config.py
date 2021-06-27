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
        'use_3x3_for_z': False,
        'use_3x3_for_output': False
    }
}


def load_json_cfg(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)
