import numpy as np
import torch
from torch import nn
from arch2vec.models.model import Model

from info_nas.models.utils import save_model_data


class Arch2vecPreprocessor:
    def __init__(self, parse_func=None, convert_back_func=None):
        self.parse_func = parse_nasbench101 if parse_func is None else parse_func
        self.convert_back_func = convert_to_nasbench101 if convert_back_func is None else convert_back_func

    def parse_net(self, ops, adj):
        return self.parse_func(ops, adj)

    def convert_back(self, ops, adj):
        return self.convert_back_func(ops, adj)

    def preprocess(self, ops, adj):
        adj = adj + adj.triu(1).transpose(-1, -2)
        return ops, adj

    def process_reverse(self, ops, adj):
        return ops, adj.triu(1)


# TODO test jestli je to stejný jako původní pad
def parse_nasbench101(ops, adj, nb_rows=7):
    if adj.shape != (nb_rows, nb_rows):
        adj = np.pad(adj, ((0, nb_rows - adj.shape[0]), (0, nb_rows - adj.shape[1])), 'constant', constant_values=0)

    transform_dict = {'input': 0, 'conv1x1-bn-relu': 1, 'conv3x3-bn-relu': 2, 'maxpool3x3': 3, 'output': 4}
    ops_array = np.zeros([nb_rows, 5], dtype='int8')
    for row, op in enumerate(ops):
        col = transform_dict[op]
        ops_array[row, col] = 1

    return ops_array, adj


def convert_to_nasbench101(ops, adj):
    transform_dict = {0: 'input', 1: 'conv1x1-bn-relu', 2: 'conv3x3-bn-relu', 3: 'maxpool3x3', 4: 'output'}
    out_ops = []
    for idx in ops:
        out_ops.append(transform_dict[idx.item()])

    adj = (adj > 0.5).int().triu(1).numpy()
    return out_ops, adj


class Arch2vecModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, latent_dim=16, num_layers=5, num_mlps=2, dropout=0.3,
                 activation_adj=torch.sigmoid, activation_ops=torch.softmax, adj_hidden_dim=128, ops_hidden_dim=128):
        super().__init__()

        self.model_kwargs = {
            'input_dim': input_dim, 'hidden_dim': hidden_dim, 'latent_dim': latent_dim, 'num_hops': num_layers,
            'num_mlp_layers': num_mlps, 'dropout': dropout, 'activation_adj': activation_adj,
            'activation_ops': activation_ops, 'adj_hidden_dim': adj_hidden_dim, 'ops_hidden_dim': ops_hidden_dim,
            'return_z': True
        }

        self.model = Model(**self.model_kwargs)

    def forward(self, ops, adj):
        out = self.model.forward(ops, adj)
        return out[:-1], out[-1]

    def save_model_data(self, data=None, save_state_dict=True):
        return save_model_data(self, kwargs=self.model_kwargs, data=data, save_state_dict=save_state_dict)

    @property
    def latent_dim(self):
        return self.model.latent_dim
