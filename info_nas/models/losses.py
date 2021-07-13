import torch.nn as nn


class WeightedMSELoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, inputs, targets):
        return self.alpha * self.mse.forward(inputs, targets)


losses_dict = {
    'MSE_weighted': WeightedMSELoss,
    'MSE': nn.MSELoss(),  # TODO call it in trainer (along with loss kwargs)
    'L1': nn.L1Loss(),
    'Huber': nn.SmoothL1Loss()
}
