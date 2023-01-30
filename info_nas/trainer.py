import os.path
from copy import copy

import pytorch_lightning as pl
import torch

from info_nas.models.utils import save_model_data


class NetworkVAE(pl.LightningModule):
    def __init__(self, model, loss, preprocessor, train_metrics=None, valid_metrics=None, test_metrics=None):
        super().__init__()
        self.model = model
        self.loss = _init_loss(loss)
        self.metrics = {'train': train_metrics, 'val': valid_metrics, 'test': test_metrics}
        self.preprocessor = preprocessor

    def save_model_args(self, dir_path):
        _save_model(dir_path, self.model, 'model')

    def training_step(self, batch, batch_idx):
        return None
        #return self._step(batch, batch_idx, 'train', prog_bar=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return None
        #return self._step(batch, batch_idx, 'val', dataloader_idx=dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self._step(batch, batch_idx, 'test', dataloader_idx=dataloader_idx)

    def _step(self, batch, batch_idx, dataset_name, dataloader_idx=None, prog_bar=False):
        ops, adj = self._process_batch(batch)

        pred = self.model(ops, adj)
        loss = self.loss[dataset_name](pred, (ops, adj))

        self.log(f'{dataset_name}/loss', loss, prog_bar=prog_bar)
        self._eval_metrics(pred, (ops, adj), self.metrics[dataset_name], dataset_name)

        return loss

    def _eval_metrics(self, pred, true, metrics, prefix):
        if metrics is None:
            return

        def eval_log(name, m):
            res = m(pred, true)
            if isinstance(res, dict):
                for k, v in res.items():
                    self.log(f"{prefix}/{m_name}_{k}", v)
            else:
                self.log(f"{prefix}/{name}", res)

        for m_name, metric in metrics.items():
            eval_log(m_name, metric)

    def _process_batch(self, batch):
        ops, adj = batch['ops'], batch['adj']
        ops, adj = self.preprocessor.preprocess(ops, adj)

        return ops, adj

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
        return optimizer


class InfoNAS(NetworkVAE):
    def __init__(self, model, labeled_model, loss, labeled_loss, preprocessor, train_metrics=None,
                 labeled_train_metrics=None, valid_metrics=None, labeled_valid_metrics=None, test_metrics=None,
                 labeled_test_metrics=None):

        super().__init__(model, loss, preprocessor, train_metrics=train_metrics, valid_metrics=valid_metrics,
                         test_metrics=test_metrics)

        self.labeled_model = labeled_model
        self.labeled_loss = _init_loss(labeled_loss)
        self.labeled_metrics = {
            'train': labeled_train_metrics, 'val': labeled_valid_metrics, 'test': labeled_test_metrics
        }

    def save_model_args(self, dir_path):
        super().save_model_args(dir_path)
        _save_model(dir_path, self.labeled_model, 'labeled_model')

    def _process_batch(self, batch):
        ops, adj = super()._process_batch(batch)

        return ops, adj, batch['inputs'], batch['outputs']

    def _step(self, batch, batch_idx, dataset_name, dataloader_idx=None, prog_bar=False):
        if not self.only_labeled:
            batch, unlabeled_batch = batch['labeled'], batch['unlabeled']
            loss = super()._step(unlabeled_batch, batch_idx, dataset_name, prog_bar=prog_bar)
            self.log(f'{dataset_name}/unlabeled_loss', loss)

        ops, adj, inputs, outputs = self._process_batch(batch)
        pred_vae, z = self.model(ops, adj)
        pred_io = self.labeled_model(z, inputs=inputs)

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
    return {'train': loss, 'val': copy(loss), 'test': copy(loss)}


def _save_model(dir_path, model, model_name):
    data = save_model_data(model, save_state_dict=False)
    torch.save(data, os.path.join(dir_path, f'{model_name}_args.pt'))


def save_to_trainer_path(trainer: pl.Trainer, model):
    dir_path = trainer.logger.log_dir
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    model.save_model_args(dir_path)
