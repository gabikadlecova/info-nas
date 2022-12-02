import copy
from abc import abstractmethod

import torch.nn as nn

from info_nas.models.utils import save_model_data, import_and_init_model


class ExtendedVAEModel(nn.Module):
    def __init__(self, vae_model):
        super().__init__()
        self.vae_model = vae_model

    def get_vae(self):
        return self.vae_model

    def clone_vae(self):
        return copy.deepcopy(self.vae_model)

    @abstractmethod
    def extended_forward(self, z, **kwargs):
        pass

    def forward(self, ops, adj, **kwargs):
        vae_out, z = self.vae_model.forward(ops, adj, return_z=True)
        outputs = self.extended_forward(z, **kwargs)

        return vae_out, outputs

    def save_model_data(self, data=None):
        vae_data = self.vae_model.save_model_data(data=data, save_state_dict=False)

        for key in ['class_name', 'class_package', 'kwargs']:
            vae_data[f'vae_{key}'] = vae_data[key]
            vae_data.pop(key)

        return save_model_data(self, kwargs=self.model_kwargs, data=vae_data, save_state_dict=True)


def load_model_from_data(data):
    load_params = ['class_name', 'class_package', 'kwargs']
    vae_package, vae_name, vae_kwargs = [data[f"vae_{p}"] for p in load_params]
    package, name, kwargs = [data[p] for p in load_params]

    vae_model = import_and_init_model(vae_name, vae_package, vae_kwargs)
    model = import_and_init_model(name, package, kwargs, vae_model, state_dict=data['state_dict'])

    return model
