import torch
from torch import nn

from info_nas.logging.base import BaseLogger
from info_nas.metrics.base import BaseMetric, MeanMetric, MetricList, SimpleMetric


class VAETrainer:
    def __init__(self, model, optimizer, preprocessor, loss, metrics, logger: BaseLogger = None,
                 verbose=1, device=None, clip=5):

        self.logger = logger if logger is not None else BaseLogger(verbose=verbose)

        self.unlabeled_loss = init_loss(loss, name='unlabeled_loss')
        self.unlabeled_metrics = init_metrics(metrics, name='unlabeled_metrics')

        self.preprocessor = preprocessor
        self.model = model
        self.optimizer = optimizer

        self.verbose = verbose  # TODO add tqdm
        self.device = device
        self.clip = clip

    def train(self, model, train_data, validation_data=None, n_epochs=1):
        model = model.to(self.device)

        for epoch in range(n_epochs):
            self.logger.log_message(f"Epoch {epoch}")
            self.epoch_start()
            model.train()

            for i_batch, batch in enumerate(train_data):
                self.optimizer.zero_grad()

                batch = self.process_batch(batch)
                loss = self.train_on_batch(model, batch, epoch, i_batch)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.clip)
                self.optimizer.step()

            if validation_data is not None:
                # eval on one or multiple validation sets
                if isinstance(validation_data, dict):
                    self.logger.log_message("Evaluate on validation sets.")

                    for val_key, val_set in validation_data.items():
                        self.logger.log_message(f"Val set: {val_key}", priority=2)
                        self.eval_validation(model, val_set, epoch, val_name=val_key)
                else:
                    self.logger.log_message("Evaluate on the validation set.")
                    self.eval_validation(model, validation_data, epoch)

            # TODO checkpointing

            self.epoch_end(epoch)

        self.logger.log_message("End of training.", priority=2)

    def train_on_batch(self, model, batch, epoch, i_batch):
        ops, adj = batch
        pred = model(ops, adj)
        loss = self.unlabeled_loss.next_batch((ops, adj), pred)

        self.logger.log_batch_metric(self.unlabeled_loss.name, loss.detach().item(), epoch, i_batch)

        return loss

    def process_batch(self, batch):
        ops, adj = batch['adj'].to(self.device), batch['ops'].to(self.device)
        ops, adj = self.preprocessor.preprocess(ops, adj)

        return ops, adj.to(torch.long)

    def eval_validation_batch(self, model, batch):
        ops, adj = batch
        pred = model(ops, adj)
        self.unlabeled_metrics.next_batch((ops, adj), pred)

    def eval_validation(self, model, validation_set, epoch, val_name='val'):
        model.eval()
        self.unlabeled_metrics.epoch_start()

        for batch in validation_set:
            ops, adj = self.process_batch(batch)
            self.eval_validation_batch(model, (ops, adj))

        val_metrics = self.unlabeled_metrics.epoch_end()
        self.logger.log_epoch_metric(f"{val_name}_{self.unlabeled_metrics.name}", val_metrics, epoch)

    def epoch_start(self):
        self.unlabeled_loss.epoch_start()

    def epoch_end(self, epoch):
        final_loss = self.unlabeled_loss.epoch_end()
        self.logger.log_epoch_metric(self.unlabeled_loss.name, final_loss, epoch)


class IOTrainer(VAETrainer):
    def __init__(self, model, optimizer, preprocessor, loss, metrics, labeled_loss, labeled_metrics, verbose=True,
                 device=None, clip=5):

        super().__init__(model, optimizer, preprocessor, loss, metrics, verbose=verbose, device=device, clip=clip)

        self.labeled_loss = init_loss(labeled_loss, name='labeled_loss')
        self.labeled_metrics = init_metrics(labeled_metrics, name='labeled_metrics')

    def train_on_batch(self, model, batch, epoch, i_batch):
        batch, is_labeled = batch
        if is_labeled:
            return super().train_on_batch(model.vae_model(), batch, epoch, i_batch)
        else:
            return self.train_on_batch_labeled(model, batch, epoch, i_batch)

    def train_on_batch_labeled(self, io_model, batch, epoch, i_batch):
        ops, adj, inputs, outputs = batch
        pred_vae, pred_io = io_model(ops, adj, inputs=inputs)

        loss = self.unlabeled_loss.next_batch((ops, adj), pred_vae)
        labeled_loss = self.labeled_loss.next_batch(outputs, pred_io)

        self.logger.log_batch_metric(self.unlabeled_loss.name, loss.detach().item(), epoch, i_batch)
        self.logger.log_batch_metric(self.labeled_loss.name, labeled_loss.detach().item(), epoch, i_batch)

        loss += labeled_loss
        return loss

    def eval_validation_batch(self, model, batch):
        ops, adj, inputs, outputs = batch
        pred_vae, pred_io = model(ops, adj, inputs=inputs)
        self.labeled_metrics.next_batch(outputs, pred_io)
        self.unlabeled_metrics.next_batch((ops, adj), pred_vae)

    def eval_validation(self, model, validation_set, epoch, val_name='validation'):
        model.eval()
        self.unlabeled_metrics.epoch_start()
        self.labeled_metrics.epoch_start()

        for batch in validation_set:
            batch, is_labeled = self.process_batch(batch)

            if is_labeled:
                self.eval_validation_batch(model, batch)
            else:
                super().eval_validation_batch(model.vae_model(), batch)

        val_metrics = self.unlabeled_metrics.epoch_end()
        labeled_val_metrics = self.labeled_metrics.epoch_end()

        self.logger.log_epoch_metric(f"{val_name}_{self.unlabeled_metrics.name}", val_metrics, epoch)
        self.logger.log_epoch_metric(f"{val_name}_{self.labeled_metrics.name}", labeled_val_metrics, epoch)

    def process_batch(self, batch):
        batch, is_labeled = batch

        ops, adj = super().process_batch(batch)
        if not is_labeled:
            return (ops, adj), is_labeled

        inputs = batch['inputs'].to(self.device) if 'inputs' in batch else None
        outputs = batch['outputs'].to(self.device)

        return (ops, adj, inputs, outputs), is_labeled

    def epoch_start(self):
        super().epoch_start()
        self.labeled_loss.epoch_start()

    def epoch_end(self, epoch):
        super().epoch_end(epoch)

        final_loss = self.labeled_loss.epoch_end()
        self.logger.log_epoch_metric(self.labeled_loss.name, final_loss, epoch)


def init_loss(loss, name=''):
    loss = loss if isinstance(loss, BaseMetric) else MeanMetric(loss, name=name)
    if not len(loss.name):
        loss.name = name
    return loss


def init_metrics(metrics, name=''):
    if isinstance(metrics, MetricList):
        pass
    elif isinstance(metrics, list):
        metrics = [(m if isinstance(m, BaseMetric) else SimpleMetric(m)) for m in metrics]
        metrics = MetricList(metrics, name=name)
    else:
        raise ValueError("Metrics must be either a list of BaseMetrics or MetricList.")

    # set names to all metrics
    if not len(name):
        name = 'metrics'
    for i, m in enumerate(metrics.metric_list):
        if not len(m.name):
            m.name = f"{name}_{i}"

    return metrics
