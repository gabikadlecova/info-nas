import torch

from info_nas.datasets.base import BaseIOExtractor, IOHook
from tqdm import tqdm


def get_nb101_graphs(nb, preprocessor, verbose=True):
    nb_net_data = {}

    if verbose:
        print("Loading nasbench101 graphs.")

    for net_hash in tqdm(nb.hash_iterator(), disable=not verbose):

        data = nb.get_metrics_from_hash(net_hash)
        ops, adj = data[0]['module_operations'], data[0]['module_adjacency']
        ops, adj = preprocessor.parse_net(ops, adj)
        nb_net_data[net_hash] = {'adj': adj, 'ops': ops}

    return nb_net_data


class NasbenchIOExtractor(BaseIOExtractor):
    def __init__(self, save_weights=True, layer_num=-1):
        self.save_weights = save_weights
        self.layer_num = layer_num

    def get_nth_layer(self, net):
        if self.layer_num > len(net.layers) + 1:
            raise ValueError(f"Layer number {self.layer_num} is larger than the number of layers in the network: "
                             f"{len(net.layers) + 1}.")
        # the last layer
        if self.layer_num == len(net.layers) or self.layer_num == -1:
            return net.classifier

        # indexing from the end is shifted by one due to net.classifier
        return net.layers[self.layer_num] if self.layer_num >= 0 else net.layers[self.layer_num + 1]

    def get_io_data(self, net, data, device=None):
        input_idx = []
        labels = []
        batch_size = None

        # register hook to get output data
        out_layer = self.get_nth_layer(net)
        hook = IOHook(save_inputs=True, save_outputs=False)
        reg_h = out_layer.register_forward_hook(hook.get_hook())

        net = net.to(device)
        net.eval()

        # fill the hook object
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data):
                inputs = inputs.to(device)
                batch_size = batch_size if batch_size is not None else len(inputs)

                net(inputs)
                input_idx.append(torch.arange(len(inputs)) + batch_idx * batch_size)
                labels.append(targets)

        reg_h.remove()

        res = {'outputs': torch.cat(hook.inputs), 'inputs': torch.cat(input_idx), 'labels': torch.cat(labels)}
        if self.save_weights:
            res['weights'] = out_layer.weight
            res['biases'] = out_layer.bias

        return res
