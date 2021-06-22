import json

local_cfg = {
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


def load_json_cfg(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)
