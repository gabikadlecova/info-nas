from info_nas.config.arch2vec import archvec_configs


INFONAS_CONFIGS = {}
INFONAS_CONFIGS.update({f"arch2vec_{k}": v for k, v in archvec_configs.items()})
