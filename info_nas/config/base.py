def get_metrics(metrics_func, key, n_loaders=1):
    if n_loaders == 0:
        return {}

    if n_loaders == 1:
        return {key: metrics_func()}

    return {f"{key}{i}": metrics_func() for i in range(n_loaders)}


def get_all_metrics(metrics_func, n_train=1, n_val=1, n_test=0, prefix=''):
    res = {}
    res.update(get_metrics(metrics_func, f'{prefix}train_', n_loaders=n_train))
    res.update(get_metrics(metrics_func, f'{prefix}val_', n_loaders=n_val))
    res.update(get_metrics(metrics_func, f'{prefix}test_', n_loaders=n_test))
    return res
