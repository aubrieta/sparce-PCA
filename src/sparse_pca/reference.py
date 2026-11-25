import numpy as np


def dense_operator(x, v):
    mu = x.mean(axis=0)
    return x.T @ (x @ v) - len(x) * mu * (mu @ v)


def sparse_operator(x, v):
    rows, cols = np.nonzero(x)
    values = x[rows, cols]
    u = np.bincount(rows, weights=values*v[cols], minlength=len(x))
    w = np.bincount(cols, weights=values*u[rows], minlength=x.shape[1])
    mu = x.mean(axis=0)
    return w-len(x)*mu*(mu@v)


def distributed_operator(parts, v):
    c = sum(map(len, parts))
    mu = sum((p.sum(axis=0) for p in parts))/c
    return sum((p.T@(p@v) for p in parts))-c*mu*(mu@v)


def initial_vector(genes, seed):
    v = np.random.default_rng(seed).normal(size=genes)
    return v/np.linalg.norm(v)


def power_iteration(x, iterations=20, seed=0, operator=dense_operator, zero_threshold=1e-12):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    v = initial_vector(x.shape[1], seed)
    for _ in range(iterations):
        w = operator(x, v)
        norm = np.linalg.norm(w)
        if norm > zero_threshold:
            v = w/norm
    av = operator(x,v)
    eigenvalue = float(v@av/((len(x)-1)*(v@v)))
    return v, eigenvalue


def oracle(x):
    z = x-x.mean(axis=0)
    _, s, vt = np.linalg.svd(z, full_matrices=False)
    return vt[0], float(s[0]**2/(len(x)-1))
