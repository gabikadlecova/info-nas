import click
import os

import torch
from arch2vec.extensions.get_nasbench101_model import get_arch2vec_model
from arch2vec.models.configs import configs
from arch2vec.utils import load_json, preprocessing
from info_nas.models.accuracy_model import AccuracyModel

from info_nas.models.utils import load_extended_vae


def get_adj_ops_preds(model, adj, ops, is_arch2vec, cfg):
    adj = torch.vstack(adj)
    ops = torch.vstack(ops)
    adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])

    x, _ = model._encoder(ops, adj) if is_arch2vec else model.vae_model._encoder(ops, adj)
    if not is_arch2vec:
        x = model.process_z.process_z[0](x)
        x = model.process_z.process_z[1](x)
    return x


@click.command()
@click.option('--data_path', default='../data/nb_dataset.json')
@click.option('--model_path')
@click.option('--save_path', default=None)
@click.option('--config', default=4)
@click.option('--batch_size', default=256)
@click.option('--is_accuracy/--is_infonas', default=False, is_flag=True)
@click.option('--is_arch2vec/--is_infonas', default=False, is_flag=True)
def extract(data_path, model_path, save_path, config, batch_size, is_accuracy, is_arch2vec):
    if save_path is None:
        dirname, modelname = os.path.split(model_path)
        save_path = os.path.join(dirname, f"embedding_{modelname}")

    device = torch.device('cuda')
    dataset = load_json(data_path)
    vae_model, _ = get_arch2vec_model(device=device)
    if not is_arch2vec:
        args = [vae_model, 3, 10] if not is_accuracy else [vae_model]
        model, _ = load_extended_vae(model_path, args, device=device, daclass=AccuracyModel if is_accuracy else None)
    else:
        model = vae_model
        model.load_state_dict(torch.load(model_path)['model_state'])

    cfg = configs[config]

    embedding = {}
    model.eval()
    with torch.no_grad():
        print("length of the dataset: {}".format(len(dataset)))

        if os.path.exists(save_path):
            print('{} is already saved'.format(save_path))
            exit()

        print('save to {}'.format(save_path))

        batch_adj = []
        batch_ops = []
        batch_ind = []
        for ind in range(len(dataset)):
            if ind % 1000 == 0:
                print(ind)

            adj = torch.Tensor(dataset[str(ind)]['module_adjacency']).unsqueeze(0).cuda()
            ops = torch.Tensor(dataset[str(ind)]['module_operations']).unsqueeze(0).cuda()
            batch_adj.append(adj)
            batch_ops.append(ops)
            batch_ind.append(ind)

            # fill up the batch
            if len(batch_adj) < batch_size and ind != len(dataset) - 1:
                continue

            x = get_adj_ops_preds(model, batch_adj, batch_ops, is_arch2vec, cfg)
            batch_adj = []
            batch_ops = []

            for feat, i in zip(x, batch_ind):
                if is_arch2vec:
                    feat = feat.mean(dim=0)

                test_acc = dataset[str(i)]['test_accuracy']
                valid_acc = dataset[str(i)]['validation_accuracy']
                time = dataset[str(i)]['training_time']

                net_hash = dataset[str(i)]['hash']

                embedding[i] = {
                    'hash': net_hash,
                    'feature': feat.squeeze(0).cpu(), 'valid_accuracy': float(valid_acc),
                    'test_accuracy': float(test_acc), 'time': float(time)
                }
            batch_ind = []

    torch.save(embedding, save_path)
    print("finish arch2vec extraction")


if __name__ == "__main__":
    extract()
