import warnings

import torch.nn as nn

from info_nas.models.base import ExtendedVAEModel
from info_nas.models.layers import LatentNodesFlatten, get_dense_list
from info_nas.models.utils import save_model_data


class AccuracyModel(ExtendedVAEModel):
    def __init__(self, vae_model, is_log_accuracy=False, z_hidden=16, n_dense=1, n_hidden=512, dropout=None):
        super().__init__(vae_model)

        self.model_kwargs = locals()
        self.model_kwargs.pop('self')
        self.model_kwargs.pop('vae_model')

        self.process_z = LatentNodesFlatten(self.vae_model.latent_dim, z_hidden=z_hidden)

        self.first_dense = nn.Linear(z_hidden, n_hidden)
        self.dense_list = get_dense_list(n_dense, dropout, n_hidden, 1)

        self.activation = None if is_log_accuracy else nn.Sigmoid()

    def extended_forward(self, z, inputs=None):
        if inputs is not None:
            warnings.warn("Inputs are ignored in the accuracy model.")

        z = self.process_z(z)
        z = self.first_dense(z)
        z = self.dense_list(z)

        if self.activation is not None:
            z = self.activation(z)

        return z.flatten()
