from abc import abstractmethod


class BaseMetric:
    @abstractmethod
    def epoch_start(self):
        pass

    @abstractmethod
    def next_batch(self, y_true, y_pred):
        pass

    @abstractmethod
    def epoch_end(self):
        pass


# TODO logging (e.g. wandb)
class SimpleMeanMetric(BaseMetric):
    def __init__(self, loss, name='', pred_first=True):
        self.loss = loss
        self.mean_loss = MetricMean(name)
        self.pred_first = pred_first

    def epoch_start(self):
        self.mean_loss.reset()

    def next_batch(self, y_true, y_pred):
        res = self.loss(y_pred, y_true) if self.pred_first else self.loss(y_true, y_pred)
        self.mean_loss.add(res)
        return res

    def epoch_end(self):
        pass


class MetricMean:
    def __init__(self, name=''):
        self.name = name
        self.sum = None
        self.n = 0
        self.mean = None

    def reset(self):
        self.sum, self.n, self.mean = None, 0, None

    def add(self, val):
        self.sum = val if self.sum is None else (self.sum + val)
        self.n += 1
        self.mean = self.sum / self.n
