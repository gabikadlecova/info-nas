import numpy as np
import torch
from nasbench import api
from nasbench.lib import graph_util

from info_nas.metrics.base import BaseMetric, OnlineMean
from info_nas.models.vae.arch2vec import Arch2vecPreprocessor


class ReconstructionAccuracyMetric(BaseMetric):
    def __init__(self, prepro: Arch2vecPreprocessor, name='reconstruction_accuracy', adj_threshold=0.5, batched=True):
        super().__init__(name=name)
        self.prepro = prepro
        self.metrics = {k: OnlineMean() for k in ['ops_accuracy', 'adj_recall', 'adj_false_pos', 'adj_accuracy']}
        self.threshold = adj_threshold
        self.batched = batched

    def epoch_start(self):
        for v in self.metrics.values():
            v.reset()

    def next_batch(self, y_pred, y_true):
        res = {}

        ops_recon, adj_recon, _, _, _ = y_pred
        ops, adj = y_true

        adj_recon, ops_recon = self.prepro.process_reverse(adj_recon, ops_recon)
        adj, ops = self.prepro.process_reverse(adj, ops)

        batch_size, adj_dim, _ = adj.shape

        res['ops_accuracy'] = ops_recon.argmax(dim=-1).eq(ops.argmax(dim=-1)).float().mean().item()
        res['adj_recall'] = adj_recon[adj.type(torch.bool)].sum().item() / adj.sum()

        triangle_div = batch_size * adj_dim * (adj_dim - 1) / 2.0
        res['adj_false_pos'] = adj_recon[(~adj.type(torch.bool)).triu(1)].sum().item() / (triangle_div - adj.sum())

        adj_recon_thre = adj_recon > self.threshold
        res['adj_accuracy'] = adj_recon_thre.eq(adj.type(torch.bool)).float().triu(1).sum().item() / triangle_div

        for k, v in res.items():
            self.metrics[k].add(v, batch_size=batch_size if self.batched else 1)

        return res

    def epoch_end(self):
        pass


class LatentVectorMetric(BaseMetric):
    def __init__(self, name=''):
        super().__init__(name=name)
        self.mu_list = []

    def epoch_start(self):
        self.mu_list = []

    def next_batch(self, y_true, y_pred):
        mu = y_pred[2]
        self.mu_list.append(mu.detach().cpu())
        return mu

    def epoch_end(self):
        pass


# TODO cite that it's from arch2vec
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
            adj_decode, ops_decode = self.prepro.convert_back(adj, max_idx)

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
        return validity, uniqueness  # TODO save it, return as a dict


class ValidityUniquenessNasbench101:
    def __init___(self, nasbench):
        self.nasbench = nasbench

    def __call__(self, ops, adj):
        adj_decode_list = np.ndarray.tolist(adj)
        spec = api.ModelSpec(matrix=adj_decode_list, ops=ops)

        return self.nasbench.is_valid(spec)
