#!/usr/bin/env bash
set -euo pipefail
SPCA_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$SPCA_ROOT"
export PYTHONPATH="$SPCA_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
SPCA_PYTHON=${SPCA_PYTHON:-python3}
set +e
"$SPCA_PYTHON" -m sparse_pca.cli run-sweep --out results/sparsity-pilot --mpspdz /tmp/MP-SPDZ
SPCA_STATUS=$?
set -e
"$SPCA_PYTHON" -m sparse_pca.cli summarize-sweep --out results/sparsity-pilot
"$SPCA_PYTHON" -m sparse_pca.cli bundle --out results/sparsity-pilot
exit "$SPCA_STATUS"
