import torch.nn as nn


class WeightedLoss(nn.Module):
    def __init__(self, alpha, loss):
        super().__init__()
        self.alpha = alpha

        if loss == 'weighted':
            raise ValueError("Cannot nest weighted loss.")

        self.loss = loss

    def forward(self, inputs, targets):
        return self.alpha * self.loss.forward(inputs, targets)
