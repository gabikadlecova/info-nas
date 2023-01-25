import torch

from info_nas.datasets.base import BaseIOExtractor, IOHook, BaseNetworkData
from tqdm import tqdm


class Nasbench101Data(BaseNetworkData):
    def __init__(self, nb, preprocessor, verbose=True, return_accuracy=False, acc_epoch=108,
                 acc_key='final_validation_accuracy'):
        self.nb = nb
        self.preprocessor = preprocessor
        self.verbose = verbose
        self.return_accuracy = return_accuracy
        self.acc_epoch = acc_epoch
        self.acc_key = acc_key

        self.net_data = {}

    def load(self):
        if self.verbose:
            print("Loading nasbench101 graphs.")

        for net_hash in tqdm(self.nb.hash_iterator(), disable=not self.verbose):

            data = self.nb.get_metrics_from_hash(net_hash)
            ops, adj = data[0]['module_operations'], data[0]['module_adjacency']
            ops, adj = self.preprocessor.parse_net(ops, adj)

            res = {'adj': adj, 'ops': ops}
            if self.return_accuracy:
                acc_data = data[1][self.acc_epoch]
                res[self.acc_key] = [a[self.acc_key] for a in acc_data]

            self.net_data[net_hash] = res

        return self.net_data

    def get_data(self, net_hash):
        return self.net_data[net_hash]


class Nasbench101Extractor(BaseIOExtractor):
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
            res['weights'] = out_layer.weight.detach()
            res['biases'] = out_layer.bias.detach()

        return res
