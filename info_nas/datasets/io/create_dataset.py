import numpy as np
import os
import torch
from torch import nn

from typing import List, Tuple
from nasbench_pytorch.model import Network as NBNetwork
from info_nas.datasets.networks.utils import load_trained_net


# TODO use valid set? train set?

# TODO adjacency apod dostat z arch2vecu, tam už je veškerej prepro (imputace apod)
#   - na to nějakou fci v utils
#   - teda kromě prepro co se dělá v train loopu


def load_io_dataset(dataset_path: str, device=None):
    data = torch.load(dataset_path, map_location=device)
    return data['net_hashes'], data['inputs'], data['outputs']


def dataset_from_pretrained(net_dir: str, nasbench, dataset, save_path: str, random_state=1, device=None, **kwargs):

    # pretrained networks in a folder
    net_paths = os.listdir(net_dir)
    networks = [load_trained_net(net_path, nasbench, device=device) for net_path in net_paths]

    dataset = create_io_dataset(networks, dataset, random_state=random_state, device=device, **kwargs)

    hashes, inputs, outputs = dataset
    hashes = torch.tensor(hashes)

    res = {'net_hashes': hashes, 'inputs': inputs, 'outputs': outputs}
    torch.save(res, save_path)

    return res


def create_io_dataset(networks: List[Tuple[str, NBNetwork]], dataset, nth_input=0, nth_output=-2, random_state=1,
                      loss=None, device=None):
    _, _, valid_loader, validation_size, _, _ = dataset

    net_hashes = []
    in_list = []
    out_list = []

    # get the io info per network
    for net_hash, network in networks:
        net_res = _get_net_outputs(network, valid_loader, nth_input, nth_output, loss=loss, num_data=validation_size,
                                   device=device)
        in_data, out_data = net_res["in_data"], net_res["out_data"]
        assert in_data.shape[0] == out_data.shape[0]

        for _ in range(in_data.shape[0]):
            net_hashes.append(net_hash)

        in_list.append(in_data)
        out_list.append(out_data)

    # form a shuffled dataset
    net_hashes = np.array(net_hashes)
    in_list = torch.cat(in_list)
    out_list = torch.cat(out_list)

    assert len(net_hashes) == len(in_list) and len(net_hashes) == len(out_list)

    indices = np.arange(len(net_hashes))
    state = np.random.RandomState(seed=random_state) if random_state is not None else np.random
    state.shuffle(indices)

    return net_hashes[indices], in_list[indices], out_list[indices]


def _get_net_outputs(net: NBNetwork, data_loader, nth_input, nth_output, loss=None, num_data=None, device=None):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net = net.to(device)

    net.eval()

    if loss is None:
        loss = nn.CrossEntropyLoss()

    test_loss = 0
    correct = 0

    n_tests = 0

    in_data = []
    out_data = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)

            curr_loss = loss(outputs, targets)
            test_loss += curr_loss.detach()
            _, predict = torch.max(outputs.data, 1)
            correct += predict.eq(targets.data).sum().detach()

            in_list, out_list = net.get_cell_outputs(inputs, return_inputs=True)

            in_data.append(in_list[nth_input])
            out_data.append(out_list[nth_output])

            if num_data is None:
                n_tests += len(targets)

        if num_data is None:
            num_data = n_tests

    last_loss = test_loss / len(data_loader) if len(data_loader) > 0 else np.inf
    acc = correct / num_data

    return {
        'in_data': torch.cat(in_data),
        'out_data': torch.cat(out_data),

        'loss': last_loss,
        'accuracy': acc
    }
