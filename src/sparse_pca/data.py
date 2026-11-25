import hashlib
import json
from pathlib import Path
import numpy as np
from .synthetic import generate


def partition(x, fractions, seed):
    fractions = np.asarray(fractions, dtype=float)
    if len(fractions) != 3 or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("three positive partition fractions must sum to one")
    sizes = np.floor(len(x) * fractions).astype(int)
    sizes[-1] += len(x) - sizes.sum()
    if np.any(sizes == 0):
        raise ValueError("each hospital must contain at least one cell")
    order = np.random.default_rng(seed).permutation(len(x))
    return list(np.split(x[order], np.cumsum(sizes)[:-1]))


def pbmc(config):
    try:
        import scanpy as sc
    except ImportError as exc:
        raise RuntimeError("PBMC preparation requires the biology extra") from exc
    a = sc.datasets.pbmc3k()
    sc.pp.filter_cells(a, min_genes=200)
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.normalize_total(a, target_sum=10000)
    sc.pp.log1p(a)
    ids = np.asarray(a.var["gene_ids"].astype(str))
    order = np.argsort(ids, kind="stable")[:config["genes"]]
    x = a[:, order].X
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    if np.any(x.sum(axis=1) == 0):
        raise ValueError("selected genes leave empty cells; select a larger gene set")
    return x.astype(float), ids[order].tolist()


def prepare(config, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if config.get("source", "synthetic") == "pbmc3k":
        x, genes = pbmc(config)
    else:
        x = generate(config["cells"], config["genes"], config["density"],
                     config["seed"], config.get("support", "uniform"),
                     config.get("value_bound", 4.0))
        genes = [f"gene-{i}" for i in range(x.shape[1])]
    parts = partition(x, config.get("fractions", [1/3]*3), config["seed"])
    capacity = config["capacity_per_hospital"]
    if any(np.count_nonzero(p) > capacity for p in parts):
        raise ValueError("configured public capacity exceeded; no automatic data-dependent expansion")
    np.savez(destination / "input.npz", **{f"party{i}": p for i,p in enumerate(parts)})
    meta = {"config": config, "shape": list(x.shape), "genes": genes,
            "nnz": int(np.count_nonzero(x)), "density": float(np.count_nonzero(x)/x.size),
            "party_rows": [len(p) for p in parts],
            "party_nnz": [int(np.count_nonzero(p)) for p in parts],
            "max_row_nnz": int(np.count_nonzero(x, axis=1).max()),
            "padded_entries": 3*capacity,
            "input_sha256": hashlib.sha256(np.vstack(parts).astype('<f8').tobytes()).hexdigest()}
    (destination / "manifest.json").write_text(json.dumps(meta, indent=2))
    return meta


def load(directory):
    with np.load(Path(directory)/"input.npz", allow_pickle=False) as f:
        parts = [f[f"party{i}"] for i in range(3)]
    return parts, json.loads((Path(directory)/"manifest.json").read_text())
