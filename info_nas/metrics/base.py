from abc import abstractmethod
from typing import Dict, Callable, List

import numpy as np


class BaseMetric:
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def epoch_start(self):
        pass

    @abstractmethod
    def next_batch(self, y_pred, y_true):
        pass

    @abstractmethod
    def epoch_end(self):
        pass


class MetricList(BaseMetric):
    def __init__(self, metric_list: List[BaseMetric] = None, name=''):
        super().__init__(name=name)
        self.metric_list = metric_list if metric_list is not None else []

        self.names = [(str(i) if not len(m.name) else m.name) for i, m in enumerate(self.metric_list)]
        self.unique_names = set(self.names)
        assert len(metric_list) == len(self.unique_names), f"Metric names must be unique: {self.names}."

    def add_metric(self, metric: BaseMetric):
        assert metric.name is not None and len(metric.name), "Metric must have a name."
        assert metric.name not in self.unique_names, f"Metric names must be unique: {metric.name}, {self.unique_names}."

        self.metric_list.append(metric)
        self.names.append(metric.name)
        self.unique_names.add(metric.name)

    def epoch_start(self):
        for m in self.metric_list:
            m.epoch_start()

    def _apply_func(self, func):
        result = {}

        for name, m in zip(self.names, self.metric_list):
            m_res = func(m)
            if isinstance(m_res, dict):
                for k, r in m_res.items():
                    result[f"{name}_{k}"] = r
            else:
                result[name] = m_res
        return result

    def next_batch(self, y_pred, y_true):
        return self._apply_func(lambda m: m.next_batch(y_pred, y_true))

    def epoch_end(self):
        return self._apply_func(lambda m: m.epoch_end())


class SimpleMetric(BaseMetric):
    def __init__(self, loss_func, name='', pred_first=True, detach=True):
        self.loss_func = loss_func
        super().__init__(name)
        self.pred_first = pred_first
        self.detach = detach

    def epoch_start(self):
        pass

    def next_batch(self, y_pred, y_true):
        loss = self.loss_func(y_pred, y_true) if self.pred_first else self.loss_func(y_true, y_pred)
        return loss.detach().cpu() if self.detach else loss

    def epoch_end(self):
        pass


class MeanMetric(SimpleMetric):
    def __init__(self, loss_func, name='', pred_first=True, batched=True, detach=True):
        super().__init__(loss_func, name=name, pred_first=pred_first, detach=detach)
        self.mean_loss = OnlineMean()
        self.batched = batched

    def epoch_start(self):
        self.mean_loss.reset()

    def next_batch(self, y_pred, y_true):
        res = super().next_batch(y_pred, y_true)
        add_res = res if not self.detach else res.detach().numpy()

        batch_size = 1 if not self.batched else len(y_true)
        self.mean_loss.add(add_res, batch_size=batch_size)
        return res

    def epoch_end(self):
        return self.mean_loss.mean


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

    def next_batch(self, y_pred, y_true):
        res = super().next_batch(y_pred, y_true)
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
