import os

import torch
import torch.nn as nn

from info_nas.models.io_model import model_dict


class ConvBnRelu(nn.Module):
    """
    From NASBench-PyTorch
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()

        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_bn_relu(x)


class LatentNodesFlatten(nn.Module):
    def __init__(self, hidden_dim, n_nodes=7, z_hidden=16):
        super().__init__()

        self.process_z = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_nodes * hidden_dim, z_hidden),
            nn.ReLU()
        )

    def forward(self, z):
        return self.process_z(z)


def save_extended_vae(dir_name, model, optimizer, epoch, loss, labeled_loss, model_class, model_kwargs):
    """Saves a checkpoint."""
    # Record the state
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'labeled_loss': labeled_loss,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'model_class': model_class,
        'model_kwargs': model_kwargs
    }
    # Write the checkpoint
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    f_path = os.path.join(dir_name, f'model_{model_class}_epoch-{epoch}_loss-{loss}_labeled-{labeled_loss}.pt')
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
