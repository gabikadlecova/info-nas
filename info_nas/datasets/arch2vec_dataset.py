from arch2vec.extensions.get_nasbench101_model import get_nasbench_datasets


# TODO gen json file odtud if kwarg
def split_to_labeled(dataset, seed=1, **kwargs):
    dataset = get_nasbench_datasets(dataset, seed=seed, **kwargs)


def load_labeled():
    # TODO pro train/valid set nacti predtrenovany dle hashu labeled. Řvi pokud tam tu síťu nenajdeš.
    pass

# TODO a pak tu měj fci, co načte oba datasety a vrátí to jako (labeled, unlabeled).
#   - to se použije v traineru