import numpy as np
import pickle
import torch
import torchvision


def get_transforms(scale_path, include_bias, axis, normalize, scale_whole=False, axis_whole=None):
    transforms = []

    if include_bias:
        assert 'include_bias' in scale_path
        transforms.append(IncludeBias())

    scaler = load_scaler(scale_path, normalize, axis, include_bias)
    transforms.append(scaler)

    whole_path = after_scale_path(scale_path, axis_whole) if scale_whole else None

    transforms.append(SortByWeights(after_sort_scale=whole_path))
    transforms.append(ToTuple())
    transforms = torchvision.transforms.Compose(transforms)

    return transforms


def load_scaler(scale_path, normalize, axis, include_bias):
    per_label = 'per_label' in scale_path
    weighted = 'weighted' in scale_path
    if axis is not None:
        assert f'axis_{axis}.' in scale_path

    scaler = Scaler(normalize=normalize, per_label=per_label, axis=axis, weighted=weighted, include_bias=include_bias)
    scaler.load_fit(scale_path)
    return scaler


def after_scale_path(scale_path, axis):
    return scale_path.replace('scale-', f"whole_scale{'-axis_' + str(axis) if axis is not None else ''}-")


class IncludeBias:
    def __call__(self, item):
        output = item['output']
        output = torch.cat([output, torch.Tensor([1.0])])
        item['output'] = output
        item['include_bias'] = True
        return item


class SortByWeights:
    def __init__(self, fixed_label=None, return_top_n=None, after_sort_scale=None):
        self.fixed_label = fixed_label
        self.return_top_n = return_top_n

        if isinstance(after_sort_scale, str):
            with open(after_sort_scale, 'rb') as f:
                after_sort_scale = pickle.load(f)

        self.after_sort_scale = after_sort_scale

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

        if self.after_sort_scale is not None:
            mu, std = self.after_sort_scale['mean'], self.after_sort_scale['std']
            output = (output - mu) / std

        item['output'] = output
        return item


class Scaler:
    def __init__(self, net_scales=None, per_label=False, normalize=False, axis=None, weighted=False, include_bias=True):
        self.per_label = per_label
        self.normalize = normalize
        self.axis = axis

        self.include_bias = include_bias

        if weighted and not per_label:
            raise ValueError("Normalizing weighted data is supported only for per_label = True.")
        self.weighted = weighted

        self.net_scales = net_scales

    def fit(self, outputs, hashes, labels=None, net_repo=None, save_path=None):
        scales = self._fit_scales(outputs, hashes, labels=labels, net_repo=net_repo)
        self.net_scales = scales

        if save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(scales, f)

    def load_fit(self, load_path):
        with open(load_path, 'rb') as f:
            self.net_scales = pickle.load(f)

    def _fit_scales(self, outputs, hashes, labels=None, net_repo=None):

        if self.per_label:
            assert labels is not None, "Must provide labels if per_label=True."
            if self.weighted:
                assert net_repo is not None, "Must provide weights in net repo if weighted=True."

            fit_dict = {}

            for label in np.unique(labels):
                labelmap = labels == label

                label_hashes = hashes[labelmap]
                label_vals = outputs[labelmap]

                fit_dict[label] = self._get_scales_per_hash(label_vals, label_hashes, net_repo=net_repo,
                                                            weight_label=label)

            return fit_dict

        return self._get_scales_per_hash(outputs, hashes)

    def _get_scales_per_hash(self, values, hashes, net_repo=None, weight_label=None):
        scales = {}

        for net_hash in np.unique(hashes):
            filtered = values[hashes == net_hash]

            # multiply by label weights
            if self.weighted:
                weights = self._get_weights(net_repo[net_hash], weight_label).numpy()
                filtered *= weights

            mean, std = np.mean(filtered, axis=self.axis), np.std(filtered, axis=self.axis)
            hmax = np.max(filtered, axis=self.axis)

            scales[net_hash] = {'mean': mean, 'std': std, 'max': hmax}

        return scales

    def _get_weights(self, item, label):
        weights = item['weights'][label]

        if self.include_bias:
            bias = item['bias'][label]
            weights = torch.cat([weights, bias.unsqueeze(-1)])

        return weights

    def __call__(self, item):
        if self.net_scales is None:
            raise ValueError("The Scaler is not fitted with scale values.")

        net_hash = item['hash']
        output = item['output']
        label = item['label'].item()

        if self.weighted:
            output *= self._get_weights(item, label)

        scales = self.net_scales[net_hash] if not self.per_label else self.net_scales[label][net_hash]

        if self.normalize:
            mu, std = scales['mean'], scales['std']
            item['output'] = (output - mu) / (std + np.finfo(np.float32).eps)
        else:
            omax = scales['max']
            item['output'] = output / (omax + np.finfo(np.float32).eps)

        return item


class ToTuple:
    def __call__(self, item):
        return item['adj'], item['ops'], item['input'], item['output']
