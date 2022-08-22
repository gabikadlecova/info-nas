from searchspace_train import enumerate_trained_networks
from searchspace_train.base import BaseDataset


# TODO or have only unlabeled data here, and then have a separate class for handling architecture data (and fetch
#    labeled data on demand, if available
class IODataset:
    def __init__(self):
        # two types of io dataset - created on demand or as a whole & saved
        self.labeled_io_data = {}  # TODO if input images, will be returned from a net repo
        self.unlabeled_hashes = []

        self.network_data = []  # preprocessed - maybe initially raw nb data, then prepro. Or create on demand.
        # so we'll have a function "load net data", then apply arch2vec prepro...


def create_io_dataset(search_space: BaseDataset, dataset, save_path: str, device: str = None):
    # TODO
    #     - write function for easy forward hook usage
    #     - reconsider using normal model saving instead of torch jit
    pass
