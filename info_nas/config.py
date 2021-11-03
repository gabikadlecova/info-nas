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
        'num_epochs': 12,
        "optimizer": "SGD"
    },
    'io': {
        'nth_input': 0,
        'nth_output': -2,
        'loss': None
    }
}


local_model_cfg = {
    "model_class": "dense",
    "model_kwargs": {
        "n_steps": 2,
        "n_convs": 2,
        "n_dense": 2,
        "activation": "linear",
        "dropout": 0.5
    },
    "optimizer": {
        "name": "Adam",
        "lr": 1e-3,
        "betas": [0.9, 0.999],
        "eps": 1e-08
    },
    "scale": {
        "include_bias": True,
        "axis": None,
        "after_axis": 0,
        "normalize": True,
        "weighted": False,
        "per_label": False,
        "multiply_by_weights": True,
        "scale_whole": True
    },
    "out_channels": 513,
    "loss": "L1",
    "loss_kwargs": {},
    "loss_vae_weight": 1.0,
    "checkpoint": 5,
    "dataset_config": {
        "k": 300,
        "coef_k": 0.33,
        "repeat_unlabeled": 1,
        "n_workers": 4,
        "n_valid_workers": 0
    },
    "arch2vec_config": 4
}


def load_json_cfg(config_path, use_model_keys=False, use_dataset_keys=False):
    """
    Load a json config, optionally include some keys if missing.

    Args:
        config_path: Path to the config.
        use_model_keys: If True include values from the default model config if a key is missing.
        use_dataset_keys: If True include values from the default dataset config if a key is missing.

    Returns: The loaded config.

    """

    with open(config_path, 'r') as f:
        cfg = json.load(f)

    def add_missing(ref):
        for k, v in ref.items():
            if k not in cfg:
                cfg[k] = v

    if use_model_keys:
        add_missing(local_model_cfg)

    if use_dataset_keys:
        add_missing(local_dataset_cfg)

    return cfg
