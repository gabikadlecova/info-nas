import torch.nn as nn
from abc import ABC, abstractmethod

# TODO  2 ways
#    a) encode input into a vec
#    b) the embedding is the second actual input


class IOModel(nn.Module):
    def __init__(self, vae_model):
        super().__init__()
        self.vae_model = vae_model

    @abstractmethod
    def inputs_forward(self, z, inputs):
        return inputs

    def forward(self, ops, args, inputs):
        ops_recon, adj_recon, mu, logvar, z = self.vae_model.forward(ops, args)
        outputs = self.inputs_forward(z, inputs)

        return ops_recon, adj_recon, mu, logvar, outputs


class SimpleConvModel(IOModel):
    def __init__(self, vae_model, input_channels, output_channels, n_steps=2, n_convs=2):
        super().__init__(vae_model)

        channels = input_channels
        conv_list = []

        for _ in range(n_steps):
            for _ in range(n_convs - 1):
                conv_list.append(nn.Conv2d(channels, channels, 3))

            next_channels = channels // 2
            conv_list.append(nn.Conv2d(channels, next_channels, 3, stride=2))
            channels = next_channels

        conv_list.append(nn.Conv2d(channels, output_channels, 1))

        self.conv_list = nn.ModuleList(conv_list)
        # TODO kam se z?

    def inputs_forward(self, z, inputs):
        pass
        # TODO z shape?
