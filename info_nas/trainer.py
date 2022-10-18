import torch
from torch import nn

from info_nas.metrics.base import BaseMetric, MeanMetric, MetricList
from info_nas.models.io_model import IOModel


class VAETrainer:
    def __init__(self, model, optimizer, preprocessor, loss, metrics, verbose=True, device=None, clip=5):
        self.metric_logger = None  # TODO wandb nebo něco

        self.unlabeled_loss = init_loss(loss, name='unlabeled_loss')
        self.unlabeled_metrics = init_metrics(metrics, name='unlabeled_metrics')

        self.preprocessor = preprocessor
        self.model = model
        self.optimizer = optimizer

        self.verbose = verbose  # TODO add verbosity prints, tqdm
        self.device = device
        self.clip = clip

    def train(self, model, train_data, validation_data, n_epochs=1):
        model = model.to(self.device)

        # todo print
        for epoch in range(n_epochs):
            self.epoch_start()
            model.train()

            for batch in train_data:
                self.optimizer.zero_grad()

                batch = self.process_batch(batch)
                loss = self.train_on_batch(model, batch)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.clip)
                self.optimizer.step()

                # eval on one or multiple validation sets
                if isinstance(validation_data, dict):
                    for val_key, val_set in validation_data.items():
                        self.eval_validation(model, val_set, val_name=val_key)
                else:
                    self.eval_validation(model, validation_data)

                # TODO ref model train/eval
                # TODO checkpointing

            self.epoch_end()
        # todo print

    def train_on_batch(self, model, batch):
        ops, adj = batch
        pred = model(ops, adj)
        return self.unlabeled_loss.next_batch((ops, adj), pred)

    def process_batch(self, batch):
        ops, adj = batch['adj'].to(self.device), batch['ops'].to(self.device)
        ops, adj = self.preprocessor.preprocess(ops, adj)

        return ops, adj.to(torch.long)

    def eval_validation_batch(self, model, batch):
        ops, adj = batch
        pred = model(ops, adj)
        self.unlabeled_metrics.next_batch((ops, adj), pred)

    def eval_validation(self, model, validation_set, val_name='validation'):
        model.eval()
        self.unlabeled_metrics.epoch_start(message=val_name)

        for batch in validation_set:
            ops, adj = self.process_batch(batch)
            self.eval_validation_batch(model, (ops, adj))

        self.unlabeled_metrics.epoch_end()

    def epoch_start(self):
        self.unlabeled_loss.epoch_start()
        self.unlabeled_metrics.epoch_start()

    def epoch_end(self):
        self.unlabeled_loss.epoch_end()
        self.unlabeled_metrics.epoch_end()


class IOTrainer(VAETrainer):
    def __init__(self, model, optimizer, preprocessor, loss, metrics, labeled_loss, labeled_metrics, verbose=True,
                 device=None, clip=5):

        super().__init__(model, optimizer, preprocessor, loss, metrics, verbose=verbose, device=device, clip=clip)

        self.labeled_loss = init_loss(labeled_loss, name='labeled_loss')
        self.labeled_metrics = init_metrics(labeled_metrics, name='labeled_metrics')

    def train_on_batch(self, model, batch):
        batch, is_labeled = batch
        if is_labeled:
            return super().train_on_batch(model.vae_model(), batch)
        else:
            return self.train_on_batch_labeled(model, batch)

    def train_on_batch_labeled(self, io_model, batch):
        ops, adj, inputs, outputs = batch
        pred_vae, pred_io = io_model(ops, adj, inputs)

        loss = self.unlabeled_loss.next_batch((ops, adj), pred_vae)
        loss += self.labeled_loss.next_batch(outputs, pred_io)

        return loss

    def eval_validation_batch(self, model, batch):
        ops, adj, inputs, outputs = batch
        pred_vae, pred_io = model(ops, adj, inputs)
        self.labeled_metrics.next_batch(outputs, pred_io)
        self.unlabeled_metrics.next_batch((ops, adj), pred_vae)

    def eval_validation(self, model, validation_set, val_name='validation'):
        model.eval()
        self.unlabeled_metrics.epoch_start(message=val_name)
        self.labeled_metrics.epoch_start(message=val_name)

        for batch in validation_set:
            batch, is_labeled = self.process_batch(batch)

            if is_labeled:
                self.eval_validation_batch(model, batch)
            else:
                super().eval_validation_batch(model.vae_model(), batch)

        self.unlabeled_metrics.epoch_end()
        self.labeled_metrics.epoch_end()

    def process_batch(self, batch):
        batch, is_labeled = batch

        ops, adj = super().process_batch(batch)
        if not is_labeled:
            return (ops, adj), is_labeled

        inputs, outputs = batch['inputs'].to(self.device), batch['outputs'].to(self.device)
        return (ops, adj, inputs, outputs), is_labeled

    def epoch_start(self):
        super().epoch_start()
        self.labeled_loss.epoch_start()
        self.labeled_metrics.epoch_end()

    def epoch_end(self):
        super().epoch_end()
        self.labeled_loss.epoch_end()
        self.labeled_metrics.epoch_end()


def init_loss(loss, name=''):
    return loss if isinstance(loss, BaseMetric) else MeanMetric(loss, name=name)


def init_metrics(metrics, name=''):
    if isinstance(metrics, MetricList):
        return metrics
    elif isinstance(metrics, list):
        return MetricList(metrics, name=name)
    else:
        raise ValueError("Metrics must be either a list of BaseMetrics or MetricList.")
