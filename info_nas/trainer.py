import torch
from torch import nn

from info_nas.models.io_model import IOModel


class IOTrainer:
    def __init__(self, device=None):
        self.metric_logger = None  # TODO wandb nebo něco

        self.unlabeled_metrics = []  # TODO
        self.labeled_metrics = []

        self.preprocessor = None  # TODO arch2vec thing

        self.optimizer = None  # TODO

        self.verbose = True  # add verbosity prints, ten statusbar s divnym nazvem (tq neco?)
        self.device = device

    def train(self, io_model: IOModel, train_data, validation_data, n_epochs=1):
        io_model = io_model.to(self.device)
        vae_model = io_model.get_vae()

        # TODO metrics.epoch_start()
        for batch, is_labeled in train_data:
            self.optimizer.zero_grad()

            # místo toho train_on_batch. Od tyhle tridy odvodit ref trainera - ten bude jednou ignorovat labeled a
            # podruhy ne
            if is_labeled:
                self._train_batch_labeled(io_model, batch)
            else:
                self._train_batch_unlabeled(vae_model, batch)

            nn.utils.clip_grad_norm_(io_model.parameters(), 5)  # TODO hyperparam
            self.optimizer.step()

            # TODO ref model train/eval

            # TODO eval on val set
            # TODO metrics.end_epoch()

            # TODO checkpointing

    def _train_batch_labeled(self, io_model, batch):
        adj, ops, inputs, outputs = self._process_batch(batch, labeled=True)
        pred_vae, pred_io = io_model(ops, adj.to(torch.long), inputs)

        # TODO eval loss, eval metrics (both for un/labeled - _function)
        #     - save Z in metrics somehow
        # return loss

    def _train_batch_unlabeled(self, model, batch):
        adj, ops = self._process_batch(batch, labeled=False)
        pred = model(ops, adj.to(torch.long))

        # TODO eval unlabeled loss/metrics
        pass

    def _process_batch(self, batch, labeled=False):
        adj, ops = batch['adj'].to(self.device), batch['ops'].to(self.device)
        adj, ops = self.preprocessor.preprocess(adj, ops)

        if not labeled:
            return adj, ops

        inputs, outputs = batch['inputs'].to(self.device), batch['outputs'].to(self.device)
        return adj, ops, inputs, outputs
