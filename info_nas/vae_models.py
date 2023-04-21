import os.path
from copy import copy

import pytorch_lightning as pl
import torch

from info_nas.models.utils import save_model_data, load_model_from_data


def get_adam(params):
    return torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-08)


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
        return self.compute_loss(batch, batch_idx, 'train', prog_bar=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self.compute_loss(batch, batch_idx, 'val', dataloader_idx=dataloader_idx, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self.compute_loss(batch, batch_idx, 'test', dataloader_idx=dataloader_idx, prog_bar=True)

    def get_dataset_name(self, dataset_name, dataloader_idx=None):
        return dataset_name if dataloader_idx is None else f"{dataset_name}_{dataloader_idx}"

    def compute_loss(self, batch, batch_idx, dataset_name, dataloader_idx=None, prog_bar=False):
        ops, adj = self._process_batch(batch)

        pred = self.model(ops, adj)
        loss = self.loss[dataset_name](pred, (ops, adj))

        log_name = self.get_dataset_name(dataset_name, dataloader_idx=dataloader_idx)
        self.log(f'{log_name}/loss', loss, prog_bar=prog_bar)
        self._eval_metrics(pred, (ops, adj), self.metrics[dataset_name], log_name)

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
        return get_adam(self.parameters())

    @staticmethod
    def load_from_checkpoint_dir(checkpoint_dir, weights_name, cfg_func, nb, map_location=None, **kwargs):
        unlabeled_path = torch.load(os.path.join(checkpoint_dir, 'model_args.pt'), map_location=map_location)
        model = load_model_from_data(unlabeled_path)

        cfg = cfg_func(model, nb, **kwargs)
        vae_kwargs = ['loss', 'preprocessor', 'train_metrics', 'valid_metrics', 'test_metrics']
        vae_kwargs = {k: cfg[k] for k in vae_kwargs}

        return cfg, NetworkVAE.load_from_checkpoint(os.path.join(checkpoint_dir, 'checkpoints', weights_name),
                                                    map_location=map_location, model=model, **vae_kwargs)


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

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        batch, labeled_batch = batch['unlabeled'], batch['labeled']
        opt, labeled_opt = self.optimizers()

        # unlabeled step
        opt.zero_grad()
        loss = self.compute_loss(batch, batch_idx, 'train', prog_bar=True)
        self.log(f'train/unlabeled_loss', loss)
        self.manual_backward(loss)
        opt.step()

        # labeled step
        labeled_opt.zero_grad()
        loss = self.compute_loss(labeled_batch, batch_idx, 'train', prog_bar=True)
        self.manual_backward(loss)
        labeled_opt.step()

    def save_model_args(self, dir_path):
        super().save_model_args(dir_path)
        _save_model(dir_path, self.labeled_model, 'labeled_model')

    def _process_batch(self, batch):
        ops, adj = super()._process_batch(batch)
        if 'outputs' not in batch:
            return ops, adj

        return ops, adj, batch['inputs'], batch['outputs']

    def compute_loss(self, batch, batch_idx, dataset_name, dataloader_idx=None, prog_bar=False):
        if 'outputs' in batch:
            return self.compute_loss_labeled(batch, batch_idx, dataset_name, dataloader_idx=dataloader_idx,
                                             prog_bar=prog_bar)
        return super().compute_loss(batch, batch_idx, dataset_name, dataloader_idx=dataloader_idx, prog_bar=prog_bar)

    def compute_loss_labeled(self, batch, batch_idx, dataset_name, dataloader_idx=None, prog_bar=False):
        ops, adj, inputs, outputs = self._process_batch(batch)
        pred_vae, z = self.model(ops, adj, return_z=True)
        pred_io = self.labeled_model(z, inputs=inputs)

        unlabeled_loss = self.loss[dataset_name](pred_vae, (ops, adj))
        labeled_loss = self.labeled_loss[dataset_name](pred_io, outputs)
        loss = unlabeled_loss + labeled_loss

        logname = self.get_dataset_name(dataset_name, dataloader_idx=dataloader_idx)
        logname = f"{logname}_labeled"
        self.log(f'{logname}/loss', loss)
        self.log(f'{logname}/unlabeled_loss', unlabeled_loss, prog_bar=prog_bar)
        self.log(f'{logname}/labeled_loss', labeled_loss, prog_bar=prog_bar)

        self._eval_metrics(pred_vae, (ops, adj), self.metrics[dataset_name], logname)
        self._eval_metrics(pred_io, outputs, self.labeled_metrics[dataset_name], logname)

        return loss

    def configure_optimizers(self):
        optimizer = get_adam(self.model.parameters())
        return optimizer, get_adam(self.parameters())

    @staticmethod
    def load_from_checkpoint_dir(checkpoint_dir, weights_name, cfg_func, map_location=None, **kwargs):
        unlabeled_path = torch.load(os.path.join(checkpoint_dir, 'model_args.pt'), map_location=map_location)
        labeled_path = torch.load(os.path.join(checkpoint_dir, 'labeled_model_args.pt'), map_location=map_location)

        model, labeled_model = load_model_from_data(unlabeled_path), load_model_from_data(labeled_path)

        cfg = cfg_func(model, labeled_model=labeled_model, **kwargs)
        vae_kwargs = ['loss', 'preprocessor', 'labeled_loss', 'train_metrics',
                      'valid_metrics', 'test_metrics', 'labeled_train_metrics', 'labeled_valid_metrics',
                      'labeled_test_metrics']
        vae_kwargs = {k: cfg[k] for k in vae_kwargs if k in cfg}

        return cfg, InfoNAS.load_from_checkpoint(os.path.join(checkpoint_dir, 'checkpoints', weights_name),
                                                 map_location=map_location, model=model, labeled_model=labeled_model,
                                                 **vae_kwargs)


def _init_loss(loss):
    return {'train': loss, 'val': copy(loss), 'test': copy(loss)}


def _save_model(dir_path, model, model_name):
    data = save_model_data(model, save_state_dict=False)
    torch.save(data, os.path.join(dir_path, f'{model_name}_args.pt'))


def save_to_trainer_path(trainer: pl.Trainer, model):
    dir_path = trainer.logger.log_dir
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    model.save_model_args(dir_path)


