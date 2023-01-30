import torch.nn as nn
import warnings

from info_nas.models.utils import save_locals
from info_nas.models.layers import LatentNodesFlatten, get_dense_list


class AccuracyModel(nn.Module):
    def __init__(self, is_log_accuracy=False, z_hidden=16, latent_dim=16, n_dense=1, n_hidden=512, dropout=None):
        super().__init__()
        self.model_kwargs = save_locals(locals())

        self.process_z = LatentNodesFlatten(latent_dim, z_hidden=z_hidden)

        self.first_dense = nn.Linear(z_hidden, n_hidden)
        self.dense_list = get_dense_list(n_dense, dropout, n_hidden, 1)

        self.activation = None if is_log_accuracy else nn.Sigmoid()

    def forward(self, z, inputs=None):
        if inputs is not None:
            warnings.warn("Inputs are ignored in the accuracy model.")

        z = self.process_z(z)
        z = self.first_dense(z)
        z = self.dense_list(z)

        if self.activation is not None:
            z = self.activation(z)

        return z.flatten()
