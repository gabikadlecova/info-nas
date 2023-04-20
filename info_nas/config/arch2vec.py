from info_nas.datasets.search_spaces.nasbench101 import Nasbench101Data

from info_nas.models.vae.arch2vec import Arch2vecPreprocessor
from torch.nn import MSELoss

from info_nas.metrics.arch2vec import VAELoss, ReconstructionMetrics, ValidityUniqueness, ValidityNasbench101


def arch2vec_nb101_cfg(model, nb, train_metrics=True, val_metrics=True, test_metrics=False, labeled_model=None,
                       **kwargs):
    # loss and metrics
    loss = VAELoss()
    labeled_loss = MSELoss()

    prepro = Arch2vecPreprocessor()

    cfg = {'loss': loss, 'labeled_loss': labeled_loss, 'preprocessor': prepro}
    for name, use_it in zip(['train', 'valid', 'test'], [train_metrics, val_metrics, test_metrics]):
        cfg[f"{name}_metrics"] = _get_metrics(prepro, model, nb) if use_it else None

    cfg['network_data'] = Nasbench101Data(nb, prepro, **kwargs)

    return cfg


def _get_metrics(prepro, vae, nb):
    return {
        'vae': ReconstructionMetrics(prepro),
        'vu': ValidityUniqueness(prepro, vae, ValidityNasbench101(nb))
    }


archvec_configs = {
    'nasbench101': arch2vec_nb101_cfg
}
