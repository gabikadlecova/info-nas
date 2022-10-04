import numpy as np


# TODO test jestli je to stejný jako původní pad
def parse_nasbench101(adj, ops, nb_rows=7):
    if adj.shape != (nb_rows, nb_rows):
        adj = np.pad(adj, ((0, nb_rows - adj.shape[0]), (0, nb_rows - adj.shape[1])), 'constant', constant_values=0)

    transform_dict = {'input': 0, 'conv1x1-bn-relu': 1, 'conv3x3-bn-relu': 2, 'maxpool3x3': 3, 'output': 4}
    ops_array = np.zeros([7, 5], dtype='int8')
    for row, op in enumerate(ops):
        col = transform_dict[op]
        ops_array[row, col] = 1

    return adj, ops_array


def preprocess(adj, ops):
    adj = adj + adj.triu(1).transpose(-1, -2)
    return adj, ops


def process_reverse(adj, ops):
    return adj.triu(1), ops
