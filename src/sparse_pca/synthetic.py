import numpy as np


def generate(cells, genes, density, seed=0, support="uniform", value_bound=4.0):
    if cells < 2 or genes < 1 or not 0 <= density <= 1 or value_bound <= 0:
        raise ValueError("invalid shape, density, or value bound")
    rng = np.random.default_rng(seed)
    count = int(round(cells * genes * density))
    if support == "uniform":
        indices = rng.choice(cells * genes, count, replace=False)
    elif support in ("clustered", "uneven"):
        rows, cols = np.indices((cells, genes))
        weights = (1 + 20 * (rows < max(1, cells // 5))).astype(float)
        if support == "clustered":
            weights *= 1 + 20 * (cols < max(1, genes // 5))
        weights = weights.ravel()
        indices = rng.choice(cells * genes, count, replace=False,
                             p=weights / weights.sum())
    else:
        raise ValueError("unknown support distribution")
    x = np.zeros((cells, genes))
    x.flat[indices] = rng.uniform(0.1 * value_bound, value_bound, count)
    return x
