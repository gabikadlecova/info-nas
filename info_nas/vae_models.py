import os.path
from copy import copy

import pytorch_lightning as pl
import torch
from torch import nn

from info_nas.models.utils import save_model_data, load_model_from_data


def get_adam(params):
    return torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-08)


def _log_stepwise(m):
    if not (hasattr(m, 'epoch_only')):
        return None
    return not m.epoch_only


def _get(moduledict, key):
    if moduledict is None:
        return None

    if key not in moduledict:
        return None
    return moduledict[key]


class NetworkVAE(pl.LightningModule):
    def __init__(self, model, loss, preprocessor, metrics=None):
        super().__init__()
        self.model = model
        self.loss = _init_loss(loss)
        self.metrics = nn.ModuleDict(metrics) if metrics is not None else None

        self.preprocessor = preprocessor

    def save_model_args(self, dir_path):
        _save_model(dir_path, self.model, 'model')

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, batch_idx, 'train_', prog_bar=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self.compute_loss(batch, batch_idx, 'val', dataloader_idx=dataloader_idx, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self.compute_loss(batch, batch_idx, 'test_', dataloader_idx=dataloader_idx, prog_bar=True)

    def _compute_metrics(self, key, metrics):
        if metrics is None:
            return

        for name, mdict in metrics.items():
            if key in name:
                if mdict is None:
                    continue

                name = name if not name.endswith('_') else name[:-1]
                for metric_name, m in mdict.items():
                    if not hasattr(m, 'log_dict'):
                        continue
                    val = {f"{name}/{metric_name}_{mkey}": mval for mkey, mval in m.compute().items()}
                    self.log_dict(val)
                    m.reset()

    def on_train_epoch_end(self):
        self._compute_metrics('train', self.metrics)

    def on_validation_epoch_end(self):
        self._compute_metrics('val', self.metrics)

    def on_test_epoch_end(self):
        self._compute_metrics('test', self.metrics)

    def get_dataset_name(self, dataset_name, dataloader_idx=None):
        return dataset_name if dataloader_idx is None else f"{dataset_name}_{dataloader_idx}"

    def compute_loss(self, batch, batch_idx, dataset_name, dataloader_idx=None, prog_bar=False):
        ops, adj = self._process_batch(batch)

        pred = self.model(ops, adj)
        loss = self.loss[dataset_name](pred, (ops, adj))

        log_name = self.get_dataset_name(dataset_name, dataloader_idx=dataloader_idx)
        metrics_name = f"{log_name}_" if '_' not in log_name else log_name
        self.log(f'{log_name}/loss', loss, prog_bar=prog_bar)
        self._eval_metrics(pred, (ops, adj), _get(self.metrics, metrics_name), log_name)

        return loss

    def _eval_metrics(self, pred, true, metrics, prefix):
        if metrics is None:
            return

        def eval_log(name, m):
            on_step = _log_stepwise(m)
            if hasattr(m, 'log_dict'):
                m.update(pred, true)
                if on_step:
                    res = {f"{prefix}/{name}_{k}": v for k, v in m.compute().items()}
                    self.log_dict(res)
                return

            m(pred, true)
            self.log(f"{prefix}/{name}", m, on_step=on_step)

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
        vae_kwargs = ['loss', 'preprocessor', 'metrics']
        vae_kwargs = {k: cfg[k] for k in vae_kwargs}

        return cfg, NetworkVAE.load_from_checkpoint(os.path.join(checkpoint_dir, 'checkpoints', weights_name),
                                                    map_location=map_location, model=model, **vae_kwargs)


class InfoNAS(NetworkVAE):
    def __init__(self, model, labeled_model, loss, labeled_loss, preprocessor, metrics=None, labeled_metrics=None):

        super().__init__(model, loss, preprocessor, metrics=metrics)

        self.labeled_model = labeled_model
        self.labeled_loss = _init_loss(labeled_loss)

        self.labeled_metrics = nn.ModuleDict(labeled_metrics) if labeled_metrics is not None else None

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

        dataset_name = self.get_dataset_name(dataset_name, dataloader_idx=dataloader_idx)
        metrics_name = f"{dataset_name}_" if '_' not in dataset_name else dataset_name
        logname = f"{dataset_name}_labeled"
        self.log(f'{logname}/loss', loss)
        self.log(f'{logname}/unlabeled_loss', unlabeled_loss, prog_bar=prog_bar)
        self.log(f'{logname}/labeled_loss', labeled_loss, prog_bar=prog_bar)

        self._eval_metrics(pred_vae, (ops, adj), _get(self.metrics, metrics_name), logname)
        self._eval_metrics(pred_io, outputs, _get(self.labeled_metrics, f"labeled_{metrics_name}"), logname)

        return loss

    def on_train_epoch_end(self):
        self._compute_metrics('train', self.metrics)
        self._compute_metrics('train', self.labeled_metrics)

    def on_validation_epoch_end(self):
        self._compute_metrics('val', self.metrics)
        self._compute_metrics('val', self.labeled_metrics)

    def on_test_epoch_end(self):
        self._compute_metrics('test', self.metrics)
        self._compute_metrics('test', self.labeled_metrics)

    def configure_optimizers(self):
        optimizer = get_adam(self.model.parameters())
        return optimizer, get_adam(self.parameters())

    @staticmethod
    def load_from_checkpoint_dir(checkpoint_dir, weights_name, cfg_func, map_location=None, **kwargs):
        unlabeled_path = torch.load(os.path.join(checkpoint_dir, 'model_args.pt'), map_location=map_location)
        labeled_path = torch.load(os.path.join(checkpoint_dir, 'labeled_model_args.pt'), map_location=map_location)

        model, labeled_model = load_model_from_data(unlabeled_path), load_model_from_data(labeled_path)

        cfg = cfg_func(model, labeled_model=labeled_model, **kwargs)
        vae_kwargs = ['loss', 'preprocessor', 'labeled_loss', 'metrics', 'labeled_metrics']
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


