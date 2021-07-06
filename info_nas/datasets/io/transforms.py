import torch


class SortByWeights:
    def __init__(self, include_bias=True):
        self.include_bias = include_bias

    def __call__(self, item):
        in_vals = item[:3]
        output = item[3]

        label = item[-3]
        weights, bias = item[-2], item[-1]

        if self.include_bias:
            sort_key = torch.cat([weights, bias.unsqueeze(-1)], dim=1)
            output = torch.cat([output, [1.0]])
        else:
            sort_key = weights

        sort_key = sort_key[label]

        sort_key, indices = torch.sort(sort_key)
        output = output[indices]

        return in_vals.extend([output, label, weights, bias])


# TODO normalize