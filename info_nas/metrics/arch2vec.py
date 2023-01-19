import numpy as np
from nasbench import api
from nasbench.lib import graph_util
import torch
from torch import nn
from torchmetrics import Metric

from info_nas.models.vae.arch2vec import Arch2vecPreprocessor


class VAELoss:
    def __init__(self, adj_loss=None, ops_loss=None, w_ops=1.0, w_adj=1.0):
        self.adj_loss = nn.BCELoss() if adj_loss is None else adj_loss
        self.ops_loss = nn.BCELoss() if ops_loss is None else ops_loss
        self.w_ops = w_ops
        self.w_adj = w_adj

    def __call__(self, y_pred, y_true):
        ops, adj = y_true
        ops_recon, adj_recon, mu, logvar = y_pred

        loss_ops = self.ops_loss(ops_recon, ops)
        loss_adj = self.adj_loss(adj_recon, adj)
        loss = self.w_ops * loss_ops + self.w_adj * loss_adj

        kl_div = -0.5 / (ops.shape[0] * ops.shape[1]) * torch.mean(
            torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 2))
        return loss + kl_div


class ReconstructionMetrics(Metric):
    def __init__(self, preprocessor):
        super().__init__()
        self.add_state("ops_acc", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("adj_recall", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("adj_fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("adj_acc", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.preprocessor = preprocessor

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        ops, adj, ops_recon, adj_recon = _get_adj_ops(preds, target, self.preprocessor)
        adj_recon_thre = adj_recon > self.threshold
        triangle_div = _get_triangle_div(adj)

        self.ops_acc += ops_recon.argmax(dim=-1).eq(ops.argmax(dim=-1)).float().mean().item()
        self.adj_recall += adj_recon[adj.type(torch.bool)].sum().item() / adj.sum().item()
        self.adj_fp += (adj_recon[(~adj.type(torch.bool)).triu(1)].sum() / (triangle_div - adj.sum())).item()
        self.adj_acc += adj_recon_thre.eq(adj.type(torch.bool)).float().triu(1).sum().item() / triangle_div

        self.total += 1

    def compute(self):
        all_metrics = {
            'ops_acc': self.ops_acc, 'adj_recall': self.adj_recall, 'adj_fp': self.adj_fp, 'adj_acc': self.adj_acc
        }
        return {name: m.float() / self.total for name, m in all_metrics.items()}


def _get_adj_ops(preds, target, prepro):
    ops_recon, adj_recon, _, _, _ = preds
    ops, adj = target

    ops_recon, adj_recon = prepro.process_reverse(ops_recon, adj_recon)
    ops, adj = prepro.process_reverse(ops, adj)
    return ops, adj, ops_recon, adj_recon


def _get_triangle_div(adj):
    batch_size, adj_dim, _ = adj.shape
    return batch_size * adj_dim * (adj_dim - 1) / 2.0


# TODO cite that it's from arch2vec
class ValidityUniqueness(Metric):
    def __init__(self, preprocessor):
        super().__init__()
        self.add_state("validity", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("uniqueness", default=torch.tensor(0), dist_reduce_fx="sum")

        self.preprocessor = preprocessor

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        mu = preds[2]
        self.mu_list.append(mu.detach().cpu())
        # TODO avg and var from a stream

    def compute(self):
        pass


class ValidityUniquenessMetric(LatentVectorMetric):
    def __init__(self, model, validity_func, prepro: Arch2vecPreprocessor, name='validity_uniqueness',
                 n_latent_points=10000, device=None):
        super().__init__(name=name)
        self.model = model
        self.prepro = prepro

        self.n_latent_points = n_latent_points
        self.device = device
        self.validity_func = validity_func

    def epoch_end(self):
        return self.compute_validity_uniqueness()

    def compute_validity_uniqueness(self):
        z_vec = torch.cat(self.mu_list, dim=0)
        z_mean, z_std = z_vec.mean(0), z_vec.std(0)

        validity_counter = 0
        buckets = set()

        # try to generate from the latent space, measure uniqueness and validity
        self.model.eval()
        for _ in range(self.n_latent_points):
            z = torch.randn_like(z_mean).to(self.device)
            z = z * z_std + z_mean
            ops, adj = self.model.decoder(z.unsqueeze(0))

            # convert to search space format
            ops = ops.squeeze(0).detach().cpu()
            adj = adj.squeeze(0).detach().cpu()
            max_idx = torch.argmax(ops, dim=-1)
            ops_decode, adj_decode = self.prepro.convert_back(max_idx, adj)

            is_valid = self.validity_func(ops_decode, adj_decode)

            if is_valid:
                validity_counter += 1

                # get onehot for hashing
                one_hot = torch.zeros_like(ops)
                for i in range(one_hot.shape[0]):
                    one_hot[i][max_idx[i]] = 1
                one_hot = one_hot.numpy()

                # save net fingerprint
                fingerprint = graph_util.hash_module(adj_decode, one_hot.tolist())
                if fingerprint not in buckets:
                    buckets.add(fingerprint)

        validity = validity_counter / self.n_latent_points
        uniqueness = len(buckets) / (validity_counter + 1e-8)
        return {'validity': validity, 'uniqueness': uniqueness}


class ValidityNasbench101:
    def __init___(self, nasbench):
        self.nasbench = nasbench

    def __call__(self, ops, adj):
        adj_decode_list = np.ndarray.tolist(adj)
        spec = api.ModelSpec(matrix=adj_decode_list, ops=ops)

        return self.nasbench.is_valid(spec)
