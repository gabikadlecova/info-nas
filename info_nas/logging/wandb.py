import wandb
from info_nas.logging.base import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(self, run_name=None, save_dir=None, verbose=1, batch_step=1, **kwargs):
        super().__init__(verbose=verbose, batch_step=batch_step)

        wandb.init(name=run_name, dir=save_dir, **kwargs)

    def log_single_metric(self, name, metric, epoch, batch=None):
        res = {name: metric, "epoch": epoch}
        if batch is not None:
            res["batch"] = batch

        wandb.log(res)
