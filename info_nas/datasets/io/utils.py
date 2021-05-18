from nasbench import api


def load_nasbench(nasbench_path, include_metrics=False):
    if include_metrics:
        raise NotImplementedError("Metrics are not supported yet.")

    nasbench = api.NASBench(nasbench_path)

    data = []

    for i, h in enumerate(nasbench.hash_iterator()):
        m = nasbench.get_metrics_from_hash(h)

        ops = m[0]['module_operations']
        adjacency = m[0]['module_adjacency']

        data.append((ops, adjacency))

    return data

#TODO a pak fci, co to predtrenuje. to v create dataset uz nacte natrenovany
# mozna nakou pomocnou tridu na to, at v tom neny bordel (kde bude x, matice, natrenovana sit)