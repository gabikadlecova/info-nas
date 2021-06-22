import os


def mkdir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
