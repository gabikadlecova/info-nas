import pickle

import click
import os

import pandas as pd
import torch
from arch2vec.extensions.get_nasbench101_model import get_arch2vec_model
from arch2vec.models.configs import configs
from nasbench import api

from info_nas.config import load_json_cfg
from _old.eval_old import eval_labeled_validation

from info_nas.models.accuracy_model import AccuracyModel
from info_nas.metrics.losses import losses_dict

from info_nas.models.utils import load_extended_vae
from scripts_old.utils import experiment_transforms, get_eval_set


@click.command()
@click.option('--data_name')
@click.option('--dataset', default='../data/test_train_long.pt')
@click.option('--nasbench_path', default='../data/nasbench.pickle')
@click.option('--model_path')
@click.option('--config', default=4)
@click.option('--model_cfg', default='../configs/model_config.json')
@click.option('--batch_size', default=256)
@click.option('--loss_name', default='MSE')
@click.option('--is_accuracy/--is_infonas', default=False, is_flag=True)
@click.option('--is_arch2vec/--is_infonas', default=False, is_flag=True)
@click.option('--split_ratio', default=None, type=float)
@click.option('--use_larger_part/--use_smaller_part', default=False)
def extract(data_name, dataset, nasbench_path, model_path, config, model_cfg, batch_size, loss_name,
            is_accuracy, is_arch2vec, split_ratio, use_larger_part):

    if nasbench_path.endswith('.pickle'):
        with open(nasbench_path, 'rb') as f:
            nb = pickle.load(f)
    else:
        nb = api.NASBench(nasbench_path)

    device = torch.device('cuda')
    model_cfg = load_json_cfg(model_cfg)
    vae_model, _ = get_arch2vec_model(device=device)
    if not is_arch2vec:
        args = [vae_model, 3, 10] if not is_accuracy else [vae_model]
        model, _ = load_extended_vae(model_path, args, device=device, daclass=AccuracyModel if is_accuracy else None)
    else:
        model = vae_model
        model.load_state_dict(torch.load(model_path)['model_state'])

    loss = losses_dict[loss_name]()
    cfg = configs[config]

    model.eval()
    with torch.no_grad():
        transforms = experiment_transforms(model_cfg)
        data_loader = get_eval_set(data_name, dataset, nb, transforms, batch_size, split_ratio=split_ratio,
                                   use_larger_part=use_larger_part)

        res_test = eval_labeled_validation(model, data_loader, device, cfg, model_cfg, loss, nasbench=nb)
    res_test = pd.DataFrame([res_test])

    save_path = os.path.dirname(model_path)
    save_path = os.path.join(save_path, f"{os.path.basename(dataset)}{'_larger' if use_larger_part else ''}.csv")

    res_test.to_csv(save_path, index=False)


if __name__ == "__main__":
    extract()
