import torch
import torch.nn as nn
from abc import abstractmethod

# TODO  3 ways
#    a) encode input into a vec
#    b) the embedding is the second actual input
#    c) vae of io data first, then dense (u-)net
from info_nas.models.layers import ConvBnRelu, LatentNodesFlatten


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

        return ops_recon, adj_recon, mu, logvar, z, outputs


class ConcatConvModel(IOModel):
    def __init__(self, vae_model, input_channels, output_channels, z_hidden=16, n_steps=2, n_convs=2,
                 use_3x3_for_z=False, use_3x3_for_output=False):

        super().__init__(vae_model)

        self.process_z = LatentNodesFlatten(self.vae_model.latent_dim, z_hidden=z_hidden)

        channels = input_channels
        conv_list = []

        # handle concatenated zs
        if use_3x3_for_z:
            conv_list.append(ConvBnRelu(channels + z_hidden, channels, kernel_size=3, padding=1))
        else:
            conv_list.append(ConvBnRelu(channels + z_hidden, channels))

        for _ in range(n_steps):
            for _ in range(n_convs - 1):
                conv_list.append(ConvBnRelu(channels, channels, kernel_size=3, padding=1))

            # halve dimension, double channels
            next_channels = channels * 2
            conv_list.append(ConvBnRelu(channels, next_channels, kernel_size=3, stride=2, padding=1))
            channels = next_channels

        # output info
        if use_3x3_for_output:
            conv_list.append(nn.Conv2d(channels, output_channels, 3, padding=1))
        else:
            conv_list.append(nn.Conv2d(channels, output_channels, 1, padding=0))

        self.conv_list = nn.Sequential(*conv_list)
        self.activation = nn.Sigmoid()

    def inputs_forward(self, z, inputs):
        # process 2D latent features to a vector
        z = self.process_z(z)

        # concat as a separate channel
        z = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, inputs.shape[2], inputs.shape[3])
        in_and_z = torch.cat([inputs, z], dim=1)

        return self.activation(self.conv_list(in_and_z))


model_dict = {
    'concat': ConcatConvModel
}
