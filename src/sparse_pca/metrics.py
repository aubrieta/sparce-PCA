import numpy as np


def relative_error(actual, expected):
    return float(np.linalg.norm(np.asarray(actual)-expected)/max(np.linalg.norm(expected),1e-12))


def pc_metrics(x, v, eigenvalue, reference_vector, reference_eigenvalue):
    from .reference import dense_operator
    av = dense_operator(x,v)
    lam = float(v@av/(v@v))
    return {"alignment": float(abs(v@reference_vector)/(np.linalg.norm(v)*np.linalg.norm(reference_vector))),
            "eigenvalue_relative_error": relative_error(eigenvalue,reference_eigenvalue),
            "relative_residual": float(np.linalg.norm(av-lam*v)/max(np.linalg.norm(av),1e-12)),
            "degenerate": bool(reference_eigenvalue < 1e-12)}
