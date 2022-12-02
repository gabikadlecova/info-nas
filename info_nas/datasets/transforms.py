import os

import numpy as np
import pickle
import torch
import torchvision


def get_transforms(scale_path, include_bias, normalize, multiply_by_weights, scale_whole_path=False):
    """
    Gets all transforms for the training, loads saved fit data for scalers.

    Args:
        scale_path: Path to load per networks scales from.
        include_bias: Include bias to the output features.
        normalize: If True, normalize the network data, if false, min max scale.
        multiply_by_weights: If True, multiply the outputs by the weights of the network (according to the label).
        scale_whole_path: Path to a scale for the whole dataset.

    Returns: Transforms for the dataset.

    """
    transforms = []

    if include_bias:
        assert 'include_bias' in scale_path
        transforms.append(IncludeBias())

    if multiply_by_weights:
        transforms.append(MultByWeights(include_bias=include_bias))

    scaler = load_scaler(scale_path, normalize, include_bias)
    transforms.append(scaler)

    transforms.append(SortByWeights(after_sort_scale=scale_whole_path))
    transforms.append(ToTuple())
    transforms = torchvision.transforms.Compose(transforms)

    return transforms


def load_scaler(scale_path, normalize, include_bias):
    per_label = 'per_label' in scale_path
    weighted = 'weighted' in scale_path

    scaler = Scaler(normalize=normalize, per_label=per_label, weighted=weighted, include_bias=include_bias)
    scaler.load_fit(scale_path)
    return scaler


def after_scale_path(scale_path, axis):
    return scale_path.replace('scale-', f"whole_scale{'-axis_' + str(axis) if axis is not None else ''}-")


def get_scale_path(scale_dir, scale_name, include_bias, per_label, weighted, axis):
    return os.path.join(scale_dir,
                        f"scale-{scale_name}"
                        f"{'-include_bias' if include_bias else ''}"
                        f"{'-per_label' if per_label else ''}"
                        f"{'-weighted' if weighted else ''}"
                        f"{'-axis_' + str(axis) if axis is not None else ''}.pickle")


def get_all_scales(scale_dir, scale_config):
    scale_args = [scale_config["include_bias"], scale_config['per_label'], scale_config['weighted'],
                  scale_config['axis']]

    scale_train = get_scale_path(scale_dir, "train", *scale_args)
    scale_valid = get_scale_path(scale_dir, "valid", *scale_args)

    scale_whole = after_scale_path(scale_train, scale_config['after_axis'])

    return scale_train, scale_valid, scale_whole


class IncludeBias:
    def __call__(self, item):
        output = item['outputs']
        output = torch.cat([output, torch.Tensor([1.0])])
        item['outputs'] = output
        item['include_bias'] = True
        return item


class SortByWeights:
    """
    Sorts the outputs by weights corresponding to the inputs label.
    """
    def __init__(self, fixed_label=None, return_top_n=None, use_all_labels=False, after_sort_scale=None):
        """
        Initializes the sort transform.

        Args:
            fixed_label: Sorts using a fixed label instead.
            return_top_n: Return only top n features.
            after_sort_scale: Scale the data after sorting.
        """
        self.fixed_label = fixed_label
        self.return_top_n = return_top_n
        self.use_all_labels = use_all_labels

        if isinstance(after_sort_scale, str):
            with open(after_sort_scale, 'rb') as f:
                after_sort_scale = pickle.load(f)

        self.after_sort_scale = after_sort_scale

    def __call__(self, item):
        output = item['outputs']
        label = item['labels']
        weights, bias = item['weights'], item['biases']

        include_bias = item['include_bias'] if 'include_bias' in item else False

        if include_bias:
            sort_key = torch.cat([weights, bias.unsqueeze(-1)], dim=1)
        else:
            sort_key = weights

        if not self.use_all_labels:
            # sort by target label or one chosen
            sort_key = sort_key[label] if self.fixed_label is None else sort_key[self.fixed_label]

            sort_key, indices = torch.sort(sort_key, descending=True)
            output = output[indices].detach()

            output = output if self.return_top_n is None else output[:self.return_top_n]
        else:
            # features sorted by each label
            sort_key, indices = torch.sort(sort_key, descending=True)
            outputs_all = output[indices].detach()

            outputs_all = outputs_all if self.return_top_n is None else outputs_all[:, :self.return_top_n]
            output = outputs_all.flatten()

        if self.after_sort_scale is not None:
            mu, std = self.after_sort_scale['mean'], self.after_sort_scale['std']
            output = (output - mu) / std

        item['outputs'] = output
        return item


def get_weights(item, label, include_bias=True):
    weights = item['weights'][label]

    if include_bias:
        bias = item['biases'][label]
        weights = torch.cat([weights, bias.unsqueeze(-1)])

    return weights


class MultByWeights:
    def __init__(self, normalize_row=False):
        self.normalize_row = normalize_row

    def __call__(self, item):
        include_bias = item['include_bias'] if 'include_bias' in item else False

        label, output = item['labels'], item['outputs']
        output *= get_weights(item, label, include_bias=include_bias)
        if self.normalize_row:
            output = (output - torch.mean(output)) / torch.std(output)

        item['outputs'] = output
        return item


class Scaler:
    """
    Scales the outputs of every network.
    """
    def __init__(self, net_scales=None, per_label=False, normalize=False, axis=None, weighted=False, include_bias=True):
        """
        Initializes the scaling transform.

        Args:
            net_scales: Fitted scales for every network.
            per_label: Use separate scales for every label.
            normalize: If True, normalize, if False, min max scale.
            axis: Axis for the scaling.
            weighted: Scale the weighted data, not the original outputs.
            include_bias: Include bias when scaling and/or multiplying by weights.
        """

        self.per_label = per_label
        self.normalize = normalize
        self.axis = axis

        self.include_bias = include_bias

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

        if self.per_label or self.weighted:
            assert labels is not None, "Must provide labels if per_label=True or weighted=True."
            if self.weighted:
                assert net_repo is not None, "Must provide weights in net repo if weighted=True."

            fit_dict = {}

            for label in np.unique(labels):
                labelmap = labels == label

                label_hashes = hashes[labelmap] if self.per_label else hashes
                label_vals = outputs[labelmap] if self.per_label else outputs

                fit_dict[label] = self._get_scales_per_hash(label_vals, label_hashes, net_repo=net_repo,
                                                            weight_label=label)

            return fit_dict

        return self._get_scales_per_hash(outputs, hashes, net_repo=net_repo)

    def _get_scales_per_hash(self, values, hashes, net_repo=None, weight_label=None):
        scales = {}

        for net_hash in np.unique(hashes):
            filtered = values[hashes == net_hash]

            # multiply by label weights
            if self.weighted:
                weights = get_weights(net_repo[net_hash], weight_label, include_bias=self.include_bias).numpy()
                filtered *= weights

            mean, std = np.mean(filtered, axis=self.axis), np.std(filtered, axis=self.axis)
            hmax = np.max(filtered, axis=self.axis)

            scales[net_hash] = {'mean': mean, 'std': std, 'max': hmax}

        return scales

    def __call__(self, item):
        if self.net_scales is None:
            raise ValueError("The Scaler is not fitted with scale values.")

        net_hash = item['hash']
        output = item['outputs']
        label = item['labels'].item()

        scales = self.net_scales[label][net_hash] if self.per_label or self.weighted else self.net_scales[net_hash]

        if self.normalize:
            mu, std = scales['mean'], scales['std']
            item['outputs'] = (output - mu) / (std + np.finfo(np.float32).eps)
        else:
            omax = scales['max']
            item['outputs'] = output / (omax + np.finfo(np.float32).eps)

        return item


class ToTuple:
    """
    Convert to tuple batch instead of a dict batch.
    """
    def __call__(self, item):
        return item['adj'], item['ops'], item['input'], item['outputs']
