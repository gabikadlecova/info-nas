import torch
from torch import nn

from info_nas.models.io_model import IOModel


class VAETrainer:
    def __init__(self, model, optimizer, device=None, clip=5):
        self.metric_logger = None  # TODO wandb nebo něco

        self.unlabeled_loss = None  # TODO pokud neni instance base metriky, zabalit
        self.unlabeled_metrics = []  # TODO

        self.preprocessor = None  # TODO arch2vec thing

        self.model = model
        self.optimizer = optimizer

        self.verbose = True  # add verbosity prints, ten statusbar s divnym nazvem (tq neco?)
        self.device = device
        self.clip = clip

    def train(self, model, train_data, validation_data, n_epochs=1):
        model = model.to(self.device)

        for batch in train_data:
            self.optimizer.zero_grad()

            batch = self.process_batch(batch)
            loss = self.train_on_batch(model, batch)

            nn.utils.clip_grad_norm_(model.parameters(), self.clip)
            self.optimizer.step()

            # TODO ref model train/eval

            # TODO eval on val set
            # TODO metrics.end_epoch()

            # TODO checkpointing

    def train_on_batch(self, model, batch):
        adj, ops = batch
        pred = model(ops, adj.to(torch.long))

    def process_batch(self, batch):
        adj, ops = batch['adj'].to(self.device), batch['ops'].to(self.device)
        adj, ops = self.preprocessor.preprocess(adj, ops)

        return adj, ops


class IOTrainer(VAETrainer):
    def __init__(self, model, optimizer, device=None, clip=5):
        super().__init__(model, optimizer, device=device, clip=clip)

        self.labeled_loss = None  # TODO
        self.labeled_metrics = []  # TODO as MetricList

    def train_on_batch(self, model, batch):
        batch, is_labeled = batch
        if is_labeled:
            return super().train_on_batch(model.vae_model(), batch)
        else:
            return self.train_on_batch_labeled(model, batch)

    def train_on_batch_labeled(self, io_model, batch):
        adj, ops, inputs, outputs = batch
        pred_vae, pred_io = io_model(ops, adj.to(torch.long), inputs)

        # TODO eval loss, eval metrics (both for un/labeled - _function)
        #     - save Z in metrics somehow
        # return loss

    def process_batch(self, batch):
        batch, is_labeled = batch

        adj, ops = super().process_batch(batch)
        if not is_labeled:
            return (adj, ops), is_labeled

        inputs, outputs = batch['inputs'].to(self.device), batch['outputs'].to(self.device)
        return (adj, ops, inputs, outputs), is_labeled
