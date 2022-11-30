from importlib import import_module


def get_class_path(obj_class):
    return obj_class.__module__, obj_class.__name__


def save_model_data(model, kwargs=None, data=None, save_state_dict=True):
    data = {} if data is None else data
    kwargs = {} if kwargs is None else kwargs
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
