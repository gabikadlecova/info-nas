import torch


class BaseLogger:
    def __init__(self, verbose=1, batch_step=1):
        self.verbose = verbose
        self.batch_step = batch_step

    def log_batch_metric(self, name, metric, epoch, batch, log_all_steps=False):
        if log_all_steps or batch % self.batch_step == 0:
            self._log_metrics(name, metric, epoch, batch=batch)

    def log_epoch_metric(self, name, metric, epoch):
        self._log_metrics(name, metric, epoch)

    def log_message(self, message, priority=1):
        if self.verbose <= priority:
            print(message)

    def _log_metrics(self, name, metric, epoch, batch=None):
        if metric is None:
            return

        if isinstance(metric, dict):
            for k, m in metric.items():
                assert len(k)
                self.log_single_metric(f"{name}_{k}", m, epoch, batch=batch)
        else:
            self.log_single_metric(name, metric, epoch, batch=batch)

    def log_single_metric(self, name, metric, epoch, batch=None):
        pass
