import numpy as np
import pickle
import torch


class IncludeBias:
    def __call__(self, item):
        output = item['output']
        output = torch.cat([output, torch.Tensor([1.0])])
        item['output'] = output
        item['include_bias'] = True
        return item


class SortByWeights:
    def __init__(self, fixed_label=None, return_top_n=None):
        self.fixed_label = fixed_label
        self.return_top_n = return_top_n

    def __call__(self, item):
        output = item['output']
        label = item['label']
        weights, bias = item['weights'], item['bias']

        include_bias = item['include_bias'] if 'include_bias' in item else False

        if include_bias:
            sort_key = torch.cat([weights, bias.unsqueeze(-1)], dim=1)
        else:
            sort_key = weights

        if self.return_top_n is None:
            # sort by target label or one chosen
            sort_key = sort_key[label] if self.fixed_label is None else sort_key[self.fixed_label]

            sort_key, indices = torch.sort(sort_key, descending=True)
            output = output[indices].detach()
        else:
            sort_key, indices = torch.sort(sort_key, descending=True)
            outputs_all = output[indices].detach()
            output = outputs_all[:, self.return_top_n].flatten()

        item['output'] = output
        return item


class Scaler:
    def __init__(self, net_scales=None, per_label=False, normalize=False, axis=None):
        self.per_label = per_label
        self.normalize = normalize
        self.axis = axis

        self.net_scales = net_scales

    def fit(self, outputs, hashes, labels=None, save_path=None):
        scales = self._fit_scales(outputs, hashes, labels=labels)
        self.net_scales = scales

        if save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(scales, f)

    def load_fit(self, load_path):
        with open(load_path, 'rb') as f:
            self.net_scales = pickle.load(f)

    def _fit_scales(self, outputs, hashes, labels=None):

        if self.per_label:
            assert labels is not None, "Must provide labels if per_label=True."
            fit_dict = {}

            for label in np.unique(labels):
                labelmap = labels == label

                label_hashes = hashes[labelmap]
                label_vals = outputs[labelmap]

                fit_dict[label] = self._get_scales_per_hash(label_vals, label_hashes)

            return fit_dict

        return self._get_scales_per_hash(outputs, hashes)

    def _get_scales_per_hash(self, values, hashes):
        scales = {}

        for net_hash in np.unique(hashes):
            filtered = values[hashes == net_hash]

            mean, std = np.mean(filtered, axis=self.axis), np.std(filtered, axis=self.axis)
            hmax = np.max(filtered, axis=self.axis)

            scales[net_hash] = {'mean': mean, 'std': std, 'max': hmax}

        return scales

    def __call__(self, item):
        if self.net_scales is None:
            raise ValueError("The Scaler is not fitted with scale values.")

        net_hash = item['hash']
        output = item['output']
        label = item['label']

        scales = self.net_scales[net_hash] if not self.per_label else self.net_scales[label][net_hash]

        if self.normalize:
            mu, std = scales['mean'], scales['std']
            item['output'] = (output - mu) / std
        else:
            omax = scales['max']
            item['output'] = output / omax

        return item


class ToTuple:
    def __call__(self, item):
        return item['adj'], item['ops'], item['input'], item['output']
