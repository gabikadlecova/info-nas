import os
import torch
from nasbench import api
from nasbench_pytorch.model import Network as NBNetwork


def load_nasbench(nasbench_path, include_metrics=False):
    if include_metrics:
        raise NotImplementedError("Metrics are not supported yet.")

    nasbench = api.NASBench(nasbench_path)

    data = []

    for _, h in enumerate(nasbench.hash_iterator()):
        m = nasbench.get_metrics_from_hash(h)

        ops = m[0]['module_operations']
        adjacency = m[0]['module_adjacency']

        data.append((h, ops, adjacency))

    return data


def save_trained_net(net_hash, net, dir_path='./checkpoints/', info=None, net_args=None, net_kwargs=None):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    checkpoint_dict = {
        'hash': net_hash,
        'model_state_dict': net.state_dict(),
        'args': net_args,
        'kwargs': net_kwargs,
        'info': info
    }

    torch.save(checkpoint_dict, os.path.join(dir_path, f'{net_hash}.tar'))


def load_trained_net(net_path, nasbench):
    checkpoint = torch.load(net_path)

    net_m = nasbench.get_metrics_from_hash(checkpoint['hash'])

    ops = net_m[0]['module_operations']
    adjacency = net_m[0]['module_adjacency']

    net = NBNetwork((ops, adjacency), *checkpoint['args'], **checkpoint['kwargs'])
    net.load_state_dict(checkpoint['model_state_dict'])

    return net, checkpoint['info']





# TODO a pak fci, co to predtrenuje. to v create dataset uz nacte natrenovany
#  mozna nakou pomocnou tridu na to, at v tom neny bordel (kde bude x, matice, natrenovana sit)