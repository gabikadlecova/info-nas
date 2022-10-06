import numpy as np
import torch
from nasbench import api
from nasbench.lib import graph_util

from info_nas.metrics.base import BaseMetric
from info_nas.models.vae.arch2vec import Arch2vecPreprocessor


# TODO accuracy, false positives etc


class ReconstructionAccuracyMetric:
    def __init__(self, prepro: Arch2vecPreprocessor):
        self.prepro = prepro
        # TODO add all as mean metric, let this inherit from BaseMetric

    def __call__(self, y_pred, y_true):
        # TODO get_nb_model 152-158 sem/do tridy - v y_true je vystup z arch2vecu, v metrikach nutny prepreverse atd
        # tj. dat z tohohle tridu, dat sem prepro, dat to callable a cpat to do ty meanmetric
        ops_recon, adj_recon = y_pred
        ops, adj = y_true



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
    def __init__(self, model, validity_func, prepro: Arch2vecPreprocessor, name='validity_uniqueness', n_latent_points=10000, device=None):
        super().__init__(name=name)
        self.model = model
        self.prepro = prepro

        self.n_latent_points = n_latent_points
        self.device = device
        self.validity_func = validity_func

    def epoch_end(self):
        self.compute_validity_uniqueness()

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
