from abc import abstractmethod
from typing import Dict, Callable

import numpy as np


class BaseMetric:
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def epoch_start(self):
        pass

    @abstractmethod
    def next_batch(self, y_true, y_pred):
        pass

    @abstractmethod
    def epoch_end(self):
        pass


class SimpleMetric(BaseMetric):
    def __init__(self, loss_func, name='', pred_first=True):
        self.loss_func = loss_func
        super().__init__(name)
        self.pred_first = pred_first

    def epoch_start(self):
        pass

    def next_batch(self, y_true, y_pred):
        res = self.loss_func(y_pred, y_true) if self.pred_first else self.loss_func(y_true, y_pred)
        return res

    def epoch_end(self):
        pass


# TODO logging (e.g. wandb)
class MeanMetric(SimpleMetric):
    def __init__(self, loss_func, name='', pred_first=True, batched=True):
        super().__init__(loss_func, name=name, pred_first=pred_first)
        self.mean_loss = OnlineMean()
        self.batched = batched

    def epoch_start(self):
        self.mean_loss.reset()

    def next_batch(self, y_true, y_pred):
        res = super().next_batch(y_true, y_pred)

        batch_size = 1 if not self.batched else len(y_true)
        self.mean_loss.add(res, batch_size=batch_size)
        return res

    def epoch_end(self):
        pass


class StatsMetric(SimpleMetric):
    def __init__(self, loss_func, name='', pred_first=True, metric_funcs: Dict[str, Callable] = None):
        super().__init__(loss_func, name=name, pred_first=pred_first)

        metric_funcs = {} if metric_funcs is None else metric_funcs
        self.metric_funcs = {'mean': np.mean, 'std': np.std, 'min': np.min, 'max': np.max, **metric_funcs}

        self.reset_data()

    def reset_data(self):
        self.metric_data = None
        self.data_list = []

    def compute_metrics(self):
        self.metric_data = {}
        for k, func in self.metric_funcs.items():
            self.metric_data[k] = self.metric_funcs[k](self.data_list)

        return self.metric_data

    def epoch_start(self):
        self.reset_data()

    def next_batch(self, y_true, y_pred):
        res = super().next_batch(y_true, y_pred)
        self.data_list.append(res)
        return res

    def epoch_end(self):
        return self.compute_metrics()


class OnlineMean:
    def __init__(self):
        self.sum = None
        self.n = 0
        self.mean = None

    def reset(self):
        self.sum, self.n, self.mean = None, 0, None

    def add(self, val, batch_size=1):
        val = val * batch_size
        self.sum = val if self.sum is None else (self.sum + val)
        self.n += batch_size
        self.mean = self.sum / self.n