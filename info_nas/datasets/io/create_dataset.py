import numpy as np
import torch
from torch import nn

from typing import List
from nasbench_pytorch.datasets.cifar10 import prepare_dataset
from nasbench_pytorch.model import Network as NBNetwork


# TODO dataset bude train-valid-test, nas zajima valid asi? train?
# TODO jako dalsi split networku na labeled-unlabeled


def create_dataset(networks: List[NBNetwork], nth_output, batch_size=32, valid_size=1000, random_state=42, loss=None,
                   device=None, **kwargs):

    datasets = prepare_dataset(batch_size, validation_size=valid_size, random_state=random_state, **kwargs)
    _, _, valid_loader, validation_size, _, _ = datasets

    for network in networks:
        net_res = _get_net_outputs(network, valid_loader, nth_output, loss=loss, num_data=validation_size,
                                   device=device)
        # TODO dodělat

    # TODO tady vzít předtrénovaný, dataset je valid set cifar, udělat outputy (a vrátit inputy). Shuffle.
    pass

# todo fce pretrain, teď je to všechno v jupyteru.


def _get_net_outputs(net: NBNetwork, data_loader, nth_output, loss=None, num_data=None, device=None):
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

            in_data.append(in_list[nth_output])
            out_data.append(out_list[nth_output])

            if num_data is None:
                n_tests += len(targets)

        if num_data is None:
            num_data = n_tests

    last_loss = test_loss / len(data_loader) if len(data_loader) > 0 else np.inf
    acc = correct / num_data

    return {
        'in_data': torch.stack(in_data),
        'out_data': torch.stack(out_data),

        'loss': last_loss,
        'accuracy': acc
    }
