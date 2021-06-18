import torch.nn as nn
from abc import ABC, abstractmethod

# TODO  2 ways
#    a) encode input into a vec
#    b) the embedding is the second actual input


class IOModel(ABC):
    def __init__(self, vae_model):
        self.vae_model = vae_model

    @abstractmethod
    def inputs_forward(self, z, inputs):
        return inputs

    def forward(self, ops, args, inputs):
        ops_recon, adj_recon, mu, logvar, z = self.vae_model.forward(ops, args)
        outputs = self.inputs_forward(z, inputs)

        return ops_recon, adj_recon, mu, logvar, outputs


class SimpleConvModel(IOModel):
    def __init__(self, vae_model, inputs_shape, outputs_shape, n_steps=3, n_convs=2):
        super().__init__(vae_model)

        conv_list = []
        # TODO outputs first?
        nn.Conv2d()

        # TODO nějakej np space pro výběr hodnot od in shape do out shape



    def inputs_forward(self, z, inputs):
        pass
