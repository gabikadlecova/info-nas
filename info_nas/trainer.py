import torch
from torch import nn

from info_nas.logging.base import BaseLogger
from info_nas.metrics.base import BaseMetric, MeanMetric, MetricList, SimpleMetric

import pytorch_lightning as pl


class NetworkVAE(pl.LightningModule):
    def __init__(self, model, loss, preprocessor, train_metrics=None, valid_metrics=None, test_metrics=None):
        super().__init__()
        self.model = model
        self.loss = _init_loss(loss)
        self.metrics = {'train': train_metrics, 'val': valid_metrics, 'test': test_metrics}
        self.preprocessor = preprocessor

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train', prog_bar=True)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        return self._step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx, dataloader_idx):
        return self._step(batch, batch_idx, 'test')

    def _step(self, batch, batch_idx, dataset_name, prog_bar=False):
        ops, adj = self._process_batch(batch)

        pred = self.model(ops, adj)
        loss = self.loss[dataset_name](pred, (ops, adj))

        self.log(f'{dataset_name}/loss', loss, prog_bar=prog_bar)
        self._eval_metrics(pred, (ops, adj), self.metrics[dataset_name], dataset_name)

        return loss

    def _eval_metrics(self, pred, true, metrics, prefix):
        def eval_log(name, m):
            res = m(pred, true)
            self.log(f"{prefix}/{name}", res)

        for m_name, metric in metrics.items():
            if isinstance(metric, dict):
                for k, v in metric.items():
                    eval_log(f"{m_name}_{k}", v)
            else:
                eval_log(m_name, metric)

    def _process_batch(self, batch):
        ops, adj = batch['ops'], batch['adj']
        ops, adj = self.preprocessor.preprocess(ops, adj)

        return ops, adj


class InfoNAS(NetworkVAE):
    def __init__(self, model, loss, labeled_loss, preprocessor, train_metrics=None, labeled_train_metrics=None,
                 valid_metrics=None, labeled_valid_metrics=None, test_metrics=None, labeled_test_metrics=None):
        unlabeled_model = model.vae_model()
        super().__init__(unlabeled_model, loss, preprocessor, train_metrics=train_metrics, valid_metrics=valid_metrics,
                         test_metrics=test_metrics)

        self.labeled_model = model
        self.labeled_loss = _init_loss(labeled_loss)
        self.labeled_metrics = {
            'train': labeled_train_metrics, 'val': labeled_valid_metrics, 'test': labeled_test_metrics
        }

    def _step(self, batch, batch_idx, dataset_name, dataloader_idx=None, prog_bar=False):
        batch, is_labeled = batch

        if not is_labeled:
            loss = super()._step(batch, batch_idx, dataset_name, prog_bar=prog_bar)
            self.log(f'{dataset_name}/unlabeled_loss', loss)

        ops, adj, inputs, outputs = batch
        pred_vae, pred_io = self.model(ops, adj, inputs=inputs)

        unlabeled_loss = self.loss[dataset_name](pred_vae, (ops, adj))
        labeled_loss = self.labeled_loss[dataset_name](pred_io, outputs)
        loss = unlabeled_loss + labeled_loss

        self.log(f'{dataset_name}/loss', loss, prog_bar=prog_bar)
        self.log(f'{dataset_name}/unlabeled_loss', unlabeled_loss)
        self.log(f'{dataset_name}/labeled_loss', labeled_loss)

        self._eval_metrics(pred_vae, (ops, adj), self.metrics[dataset_name], dataset_name)
        self._eval_metrics(pred_io, outputs, self.labeled_metrics[dataset_name], dataset_name)

        return loss


def _init_loss(loss):
    return {'train': loss, 'val': loss.copy(), 'test': loss.copy()}
