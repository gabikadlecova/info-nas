from copy import copy
from importlib import import_module


def save_locals(locals_dict, exclude=None):
    locals_dict = copy(locals_dict)
    exclude = set(exclude) if exclude is not None else set()
    exclude.add('self')
    exclude.add('__class__')

    for key in exclude:
        locals_dict.pop(key)

    return locals_dict


def get_class_path(obj_class):
    return obj_class.__module__, obj_class.__name__


def save_model_data(model, data=None, save_state_dict=True):
    data = {} if data is None else data
    kwargs = model.model_kwargs
    path, name = get_class_path(model.__class__)

    data = {**data, 'class_name': name, 'class_package': path, 'kwargs': kwargs}
    if save_state_dict:
        data['state_dict'] = model.state_dict()

    return data


def import_and_init_model(class_name, class_package, model_kwargs, *args, state_dict=None, **kwargs):
    module = import_module(class_package)
    model = module.__dict__[class_name]
    model = model(*args, **kwargs, **model_kwargs)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model


def load_model_from_data(data):
    load_params = ['class_name', 'class_package', 'kwargs']
    name, package, kwargs = [data[p] for p in load_params]
    state_dict = data['state_dict'] if 'state_dict' in data else None

    model = import_and_init_model(name, package, kwargs, state_dict=state_dict)

    return model
