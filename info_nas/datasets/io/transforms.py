import torch


class SortByWeights:
    def __init__(self, include_bias=True, fixed_label=None, return_all_features=False):
        self.include_bias = include_bias

        self.fixed_label = fixed_label
        self.return_all_features = return_all_features

    def __call__(self, item):
        in_vals = item[:3]
        output = item[3]

        label = item[4]
        weights, bias = item[5], item[6]
        other_info = item[7:]

        if self.include_bias:
            sort_key = torch.cat([weights, bias.unsqueeze(-1)], dim=1)
            output = torch.cat([output, torch.Tensor([1.0])])
        else:
            sort_key = weights

        if not self.return_all_features:
            # sort by target label or one chosen
            sort_key = sort_key[label] if self.fixed_label is None else sort_key[self.fixed_label]

        sort_key, indices = torch.sort(sort_key, descending=True)
        output = output[indices].detach()

        in_vals.extend([output, label, weights, bias])
        in_vals.extend(other_info)
        return in_vals


# TODO normalize