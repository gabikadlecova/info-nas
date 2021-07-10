import torch.nn as nn

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


def get_conv_list(n_steps, n_convs, channels):
    conv_list = []

    for _ in range(n_steps):
        for _ in range(n_convs - 1):
            conv_list.append(ConvBnRelu(channels, channels, kernel_size=3, padding=1))

        # halve dimension, double channels
        next_channels = channels * 2
        conv_list.append(ConvBnRelu(channels, next_channels, kernel_size=3, stride=2, padding=1))
        channels = next_channels

    return nn.Sequential(*conv_list), channels
