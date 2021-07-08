import os

import torch
from torch import optim

from info_nas.models.io_model import model_dict


def save_extended_vae(dir_name, model, optimizer, epoch, model_class, model_kwargs):
    """Saves a checkpoint."""
    # Record the state
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'model_class': model_class,
        'model_kwargs': model_kwargs
    }
    # Write the checkpoint
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    f_path = os.path.join(dir_name, f'model_{model_class}_epoch-{epoch}.pt')
    torch.save(checkpoint, f_path)


def load_extended_vae(model_path, model_args, device=None, optimizer=None):
    checkpoint = torch.load(model_path, map_location=device)

    kwargs = checkpoint['model_kwargs']
    model_class = model_dict[checkpoint['model_class']]
    model = model_class(*model_args, **kwargs)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    checkpoint.pop('model_state')
    checkpoint.pop('optimizer_state')
    return model, checkpoint


def get_optimizer(model, name='adam', **kwargs):
    if name.lower() == 'adam':
        optimizer = optim.Adam
    elif name.lower() == 'sgd':
        optimizer = optim.SGD
    else:
        raise ValueError(f"Unsupported optimizer name: {name}")

    return optimizer(model.parameters(), **kwargs)
