import os
from info_nas.logging.base import BaseLogger


class CSVLogger(BaseLogger):
    def __init__(self, log_dir='./logs/', verbose=1, batch_step=1, add_header=True):
        super().__init__(verbose=verbose, batch_step=batch_step)

        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        self.add_header = add_header
        self.seen_metrics = set()

    def log_single_metric(self, name, metric, epoch, batch=None):
        with open(os.path.join(self.log_dir, name), 'a+') as f:
            # write header if enabled
            if self.add_header and name not in self.seen_metrics:
                self.seen_metrics.add(name)
                row = 'epoch,batch' if batch is not None else 'epoch'
                f.write(f'{row},{name}\n')

            # write metrics
            row = f'{epoch},{batch}' if batch is not None else f'{epoch}'
            f.write(f'{row},{metric}\n')
