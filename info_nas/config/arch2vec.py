from torch import nn

from info_nas.datasets.search_spaces.nasbench101 import Nasbench101Data

from info_nas.models.vae.arch2vec import Arch2vecPreprocessor
from torch.nn import MSELoss

from info_nas.metrics.arch2vec import VAELoss, ReconstructionMetrics, ValidityUniqueness, ValidityNasbench101
from config.base import get_all_metrics


def arch2vec_nb101_cfg(model, nb, n_train=1, n_val=1, n_test=0, labeled_model=None, labeled_metric_func=None, **kwargs):
    # loss and metrics
    loss = VAELoss()
    labeled_loss = MSELoss()

    prepro = Arch2vecPreprocessor()
    metrics = get_all_metrics(lambda: _get_metrics(prepro, model, nb), n_train=n_train, n_val=n_val, n_test=n_test)

    cfg = {'loss': loss, 'labeled_loss': labeled_loss, 'preprocessor': prepro,
           'metrics': metrics}
    if labeled_model is not None:
        if labeled_metric_func is not None:
            cfg['labeled_metrics'] = get_all_metrics(labeled_metric_func, n_train=n_train, n_val=n_val, n_test=n_test)
        else:
            cfg['labeled_metrics'] = None

    cfg['network_data'] = Nasbench101Data(nb, prepro, **kwargs)

    return cfg


def _get_metrics(prepro, vae, nb):
    return nn.ModuleDict({
        'vae': ReconstructionMetrics(prepro),
        'vu': ValidityUniqueness(prepro, vae, ValidityNasbench101(nb))
    })


archvec_configs = {
    'nasbench101': arch2vec_nb101_cfg
}
