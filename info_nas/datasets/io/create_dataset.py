import numpy as np
import os
import torch
from torch import nn

from typing import List, Union
from nasbench_pytorch.model import Network as NBNetwork
from info_nas.datasets.networks.utils import load_trained_net


def load_io_dataset(dataset_path: str, device=None):
    """
    Load the saved dataset, map the data location to `device`.

    Args:
        dataset_path: Path to the checkpoint (.pt format)
        device: Device for the data.

    Returns: The loaded IO dataset

    """
    data = torch.load(dataset_path, map_location=device)
    return data


def _list_net_dir(net_dir: str):
    net_paths = os.listdir(net_dir)
    return [os.path.join(net_dir, n) for n in net_paths]


def dataset_from_pretrained(net_dir: Union[str, List[str]], nasbench, dataset, save_path: str, device=None,
                            use_test_data=False, **kwargs):
    """
    Create the IO dataset using the checkpoints of trained networks and a dataset of inputs. Save it to a directory,
    the output file format is .pt.

    Args:
        net_dir: Either a list of directories, or one directory, where the .tar network checkpoints are loaded from.
        nasbench: An instance of nasbench.api.NASBench(nb_path).
        dataset: The input dataset with the format:
            train_loader, n_train, valid_loader, n_valid, test_loader, n_test

        save_path: Path to save the checkpoint (the output should have the .pt format).
        device: The device for the neural networks (used during prediction).
        use_test_data: If True, use the test dataset, if False, use the validation set for IO dataset creation.
        **kwargs: Additional kwargs for the `create_io_dataset` function.

    Returns: The generated IO dataset.

    """

    # pretrained networks in one folder
    if isinstance(net_dir, str):
        net_dir = [net_dir]

    # join multiple folders
    net_paths = [p for nd in net_dir for p in _list_net_dir(nd) if p.endswith('.tar')]

    print(f'Creating dataset from {len(net_paths)} pretrained networks.')

    networks = (load_trained_net(net_path, nasbench, device=device) for net_path in net_paths)

    data = create_io_dataset(networks, dataset, device=device, use_test_data=use_test_data, **kwargs)
    torch.save(data, save_path)

    return data


def create_io_dataset(networks, dataset, nth_input=0, nth_output=-2, loss=None, device=None, print_frequency=20,
                      use_test_data=False, test_subset_size=20):
    """
    Create the IO dataset with the following format (N is the size of the dataset, M is the number of trained
    networks, I is the number of images):
    {
        'net_hashes': vector of hashes, length N, M unique,
        'inputs': dataset of inputs of length N, either image data or indices of images in 'dataset' (I unique),
        'outputs': dataset of outputs of length N, either a feature vector or a feature map,
        'dataset': the dataset that was used to create the IO data (length I),
        'labels': a vector of labels (length N),
        'use_reference': if True, the 'inputs' contain indices of images from 'dataset',
        'net_repo': a dict with net hash keys (M unique), where network specific data like weights or biases (of the
            last dense layer) are stored
    }

    Args:
        networks: An iterable of trained networks for prediction. Must have the function
            get_cell_outputs(inputs, return_inputs=True) that returns a list of inputs and corresponding outputs -
            the inputs to a (hidden) layer of the network and corresponding outputs.

        dataset: The dataset for creation of the IO data.
        nth_input: The index of the returned input data in the input list.
        nth_output: The index of the returned output data in the input list.
        loss: The loss to use for evaluation of predictions (default nn.CrossEntropyLoss)
        device: The device for prediction.
        print_frequency: Prints the number of processed networks every `print_frequency`.
        use_test_data: If True, use test data for prediction, if False, validation.
        test_subset_size: Use a random sample of the test set if it is too big (in batches)

    Returns: The created IO dataset.

    """

    _, _, valid_loader, validation_size, test_loader, test_size = dataset
    # test dataset
    if use_test_data:
        validation_size = None
        test_inds = np.random.choice(np.arange(len(test_loader)), size=test_subset_size, replace=False)

        loaded_dataset = [b for i, b in enumerate(test_loader) if i in test_inds]
    else:
        loaded_dataset = [b for b in valid_loader]

    net_repo = {}

    net_hashes = []
    in_list = []
    out_list = []

    # get the io info per network
    for i, (net_hash, network, _) in enumerate(networks):
        if (i % print_frequency) == 0:
            print(f"Processing network {i}: {net_hash}")

        net_res = _get_net_outputs(network, loaded_dataset, nth_input, nth_output, loss=loss, num_data=validation_size,
                                   device=device)
        in_data, out_data = net_res["in_data"], net_res["out_data"]
        assert in_data.shape[0] == out_data.shape[0]

        for _ in range(in_data.shape[0]):
            net_hashes.append(net_hash)

        in_list.append(in_data)
        out_list.append(out_data)

        out_weight = network.classifier.weight
        out_bias = network.classifier.bias

        net_repo[net_hash] = {
            'weights': out_weight.detach().cpu(),
            'bias': out_bias.detach().cpu()
        }

    return _process_output_data(net_hashes, in_list, out_list, loaded_dataset, net_repo)


def _process_output_data(net_hashes, in_list, out_list, loaded_dataset, net_repo):
    net_hashes = np.array(net_hashes)
    in_list = torch.cat(in_list)
    out_list = torch.cat(out_list)

    # concat batched dataset
    loaded_inputs, loaded_targets = [], []
    for i, t in loaded_dataset:
        loaded_inputs.append(i)
        loaded_targets.append(t)

    loaded_inputs = torch.cat(loaded_inputs)
    loaded_targets = torch.cat(loaded_targets)

    assert len(net_hashes) == len(in_list) and len(net_hashes) == len(out_list)

    use_reference = len(in_list.shape) == 1

    # io dataset with hashes, original dataset and network info for reference
    data = {'net_hashes': net_hashes, 'inputs': in_list, 'outputs': out_list, 'dataset': loaded_inputs,
            'labels': loaded_targets, 'use_reference': use_reference, 'net_repo': net_repo}

    return data


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

    batch_size = None
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if batch_size is None:
                batch_size = len(inputs)

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            curr_loss = loss(outputs, targets)
            test_loss += curr_loss.detach()
            _, predict = torch.max(outputs.data, 1)
            correct += predict.eq(targets.data).sum().detach()

            in_list, out_list = net.get_cell_outputs(inputs, return_inputs=True)

            # if first input (original image), save reference index instead
            if nth_input != 0:
                save_input = in_list[nth_input].to('cpu')
            else:
                save_input = torch.arange(len(inputs)) + batch_idx * batch_size

            in_data.append(save_input)
            out_data.append(out_list[nth_output].to('cpu'))

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
