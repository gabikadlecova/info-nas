import torch
import torch.nn as nn
from abc import abstractmethod

# TODO  3 ways
#    a) encode input into a vec
#    b) the embedding is the second actual input
#    c) vae of io data first, then dense (u-)net
from info_nas.models.layers import ConvBnRelu, LatentNodesFlatten, get_conv_list


class IOModel(nn.Module):
    """
    A model that processes architecture data (using a VAE) as well as IO data (using the VAE encoder and a regressor).
    """
    def __init__(self, vae_model, activation=None):
        super().__init__()
        self.vae_model = vae_model

        if activation is None or activation.lower() == 'linear':
            self.activation = None
        elif activation.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation")

    @abstractmethod
    def inputs_forward(self, z, inputs):
        return inputs

    def forward(self, ops, args, inputs):
        ops_recon, adj_recon, mu, logvar, z = self.vae_model.forward(ops, args)
        outputs = self.inputs_forward(z, inputs)

        return ops_recon, adj_recon, mu, logvar, z, outputs


class ConcatConvModel(IOModel):
    """
    An IO model where the network representation is concatenated to the network along the channel axis (repeated across
    spatial dimensions).
    """
    def __init__(self, vae_model, input_channels, output_channels, start_channels=128, z_hidden=16,
                 n_steps=2, n_convs=2, dense_output=True, activation=None,
                 use_3x3_for_z=False, use_3x3_for_output=False):

        super().__init__(vae_model, activation=activation)

        self.process_z = LatentNodesFlatten(self.vae_model.latent_dim, z_hidden=z_hidden)

        channels = start_channels

        # handle concatenated zs
        if use_3x3_for_z:
            self.concat_conv = ConvBnRelu(input_channels + z_hidden, channels, kernel_size=3, padding=1)
        else:
            self.concat_conv = ConvBnRelu(input_channels + z_hidden, channels)

        self.conv_list, channels = get_conv_list(n_steps, n_convs, channels)

        self.dense_output = dense_output
        # output info
        if dense_output:
            self.last_layer = nn.Linear(channels, output_channels)
        else:
            if use_3x3_for_output:
                self.last_layer = nn.Conv2d(channels, output_channels, 3, padding=1)
            else:
                self.last_layer = nn.Conv2d(channels, output_channels, 1, padding=0)

    def inputs_forward(self, z, inputs):
        # process 2D latent features to a vector
        z = self.process_z(z)

        # concat as a separate channel
        z = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, inputs.shape[2], inputs.shape[3])
        z = torch.cat([inputs, z], dim=1)

        z = self.concat_conv(z)
        z = self.conv_list(z)

        if self.dense_output:
            z = torch.mean(z, (2, 3))

        z = self.last_layer(z)
        return self.activation(z) if self.activation is not None else z


class DensePredConvModel(IOModel):
    """
    An IO model where the network and image vector representations are concatenated and then passed through dense
    layers.
    """
    def __init__(self, vae_model, input_channels, output_channels, start_channels=128, activation=None, z_hidden=16,
                 n_steps=2, n_convs=2, n_dense=1, dense_size=512, dropout=None):

        super().__init__(vae_model, activation=activation)

        # process images
        self.first_conv = ConvBnRelu(input_channels, start_channels, kernel_size=3, padding=1)
        self.conv_list, channels = get_conv_list(n_steps, n_convs, start_channels)

        # process network data
        self.process_z = LatentNodesFlatten(self.vae_model.latent_dim, z_hidden=z_hidden)
        self.concat_dense = nn.Linear(z_hidden + channels, dense_size)

        # process concatenated data
        dense_list = []

        for i in range(n_dense):
            dense_list.append(nn.ReLU())
            if dropout is not None:
                dense_list.append(nn.Dropout(dropout))

            next_size = output_channels if i == n_dense - 1 else dense_size
            dense_list.append(nn.Linear(dense_size, next_size))

        self.dense_list = nn.Sequential(*dense_list)

    def inputs_forward(self, z, inputs):
        x = self.first_conv(inputs)
        x = self.conv_list(x)
        x = torch.mean(x, (2, 3))

        z = self.process_z(z)
        z = torch.cat([z, x], dim=1)
        z = self.concat_dense(z)

        z = self.dense_list(z)
        return self.activation(z) if self.activation is not None else z


model_dict = {
    'concat': ConcatConvModel,
    'dense': DensePredConvModel
}
