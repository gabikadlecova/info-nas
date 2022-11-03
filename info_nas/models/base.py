import copy
from abc import abstractmethod

import torch.nn as nn


class ExtendedVAEModel(nn.Module):
    def __init__(self, vae_model):
        super().__init__()
        self.vae_model = vae_model

    def get_vae(self):
        return self.vae_model

    def clone_vae(self):
        return copy.deepcopy(self.vae_model)

    @abstractmethod
    def extended_forward(self, z, **kwargs):
        pass

    def forward(self, ops, adj, **kwargs):
        vae_out, z = self.vae_model.forward(ops, adj)
        outputs = self.extended_forward(z, **kwargs)

        return vae_out, outputs
