import pickle
import torch


class IncludeBias:
    def __call__(self, item):
        output = item['outputs']
        output = torch.cat([output, torch.Tensor([1.0])])
        item['outputs'] = output
        item['include_bias'] = True
        return item


def _get_weights(item):
    # include bias if included in outputs
    include_bias = item['include_bias'] if 'include_bias' in item else False
    if include_bias:
        return torch.cat([item['weights'], item['bias'].unsqueeze(-1)], dim=1)
    else:
        return item['weights']


class MultiplyByWeights:
    def __call__(self, item):
        output = item['outputs']
        weights = _get_weights(item)

        output *= weights
        item['outputs'] = weights

        return item


class SortByWeights:
    """
    Sorts the outputs by weights corresponding to the inputs label.
    """
    def __init__(self, fixed_label=None, return_top_n=None, use_all_labels=False):
        """
        Initializes the sort transform.

        Args:
            fixed_label: Sorts using a fixed label instead.
            return_top_n: Return only top n features.
            use_all_labels: Return features for all labels.
        """
        self.fixed_label = fixed_label
        self.return_top_n = return_top_n
        self.use_all_labels = use_all_labels

    def __call__(self, item):
        output = item['outputs']
        label = item['labels']
        weights = _get_weights(item)

        # sort and return outputs of the correct label/all outputs
        if not self.use_all_labels:
            # sort by target label or one chosen
            weights = weights[label] if self.fixed_label is None else weights[self.fixed_label]

            weights, indices = torch.sort(weights, descending=True)
            output = output[indices].detach()

            output = output if self.return_top_n is None else output[:self.return_top_n]
        else:
            # features sorted by each label
            weights, indices = torch.sort(weights, descending=True)
            outputs_all = output[indices].detach()

            outputs_all = outputs_all if self.return_top_n is None else outputs_all[:, :self.return_top_n]
            output = outputs_all.flatten()

        item['outputs'] = output
        return item
