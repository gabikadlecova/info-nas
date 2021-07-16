import torch.nn as nn


class WeightedLoss(nn.Module):
    def __init__(self, alpha, loss='L1'):
        super().__init__()
        self.alpha = alpha

        if loss == 'weighted':
            raise ValueError("Cannot nest weighted loss.")

        self.loss = losses_dict[loss]()

    def forward(self, inputs, targets):
        return self.alpha * self.loss.forward(inputs, targets)


losses_dict = {
    'weighted': WeightedLoss,
    'MSE': nn.MSELoss,
    'L1': nn.L1Loss,
    'Huber': nn.SmoothL1Loss
}
