from info_nas.config.arch2vec import arch2vec_configs


INFONAS_CONFIGS = {}
INFONAS_CONFIGS.update({f"arch2vec_{k}": v for k, v in arch2vec_configs.items()})
