#!/usr/bin/env bash
set -euo pipefail
SPCA_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$SPCA_ROOT"
export PYTHONPATH="$SPCA_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
SPCA_PYTHON=${SPCA_PYTHON:-python3}
# Compile must have been completed first. Always package logs, including failures.
set +e
"$SPCA_PYTHON" -m sparse_pca.cli run-suite --out results/smoke --mpspdz /tmp/MP-SPDZ
SPCA_STATUS=$?
set -e
"$SPCA_PYTHON" -m sparse_pca.cli report --out results/smoke
"$SPCA_PYTHON" -m sparse_pca.cli bundle --out results/smoke
exit "$SPCA_STATUS"
