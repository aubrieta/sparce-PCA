# Sparse, implicitly centered PCA with MP-SPDZ

This repository compares dense and sparse secure PC1 on identical horizontally
partitioned matrices. The centered operator is

`A(v) = sum_i X_i.T @ (X_i @ v) - C * mu * (mu.T @ v)`.

The sample covariance eigenvalue is obtained by dividing the Rayleigh quotient by
`C - 1`. No covariance matrix or explicitly centered expression matrix is formed.

## Current verification status

Seventeen local tests pass (the separate compiler integration test is opt-in).
All six smoke programs compiled and have passed actual three-party MPC
execution and numerical verification in the user's Terminal. Dense and
private-support results are in `results/smoke`; corrected public-support results
are in `results/public-support-fix`. The original public-support failures in
`results/smoke` are retained as debugging evidence.

The public-support retest gives operator relative error `1.24e-6`, PC1 alignment
`0.999999999998`, and relative eigenvalue error `3.53e-7`. These are small-fixture
correctness checks. Repeated performance trials, sparsity crossovers, larger
matrices, and PBMC3k execution remain unverified. Single smoke timings do not
establish performance conclusions.

## Requirements

- Python 3.11+ and NumPy. `pyproject.toml` pins the supported dependency versions.
- MP-SPDZ at `/tmp/MP-SPDZ`, including `replicated-ring-party.x` and its libraries.
- The tested compiler revision is
  `f5f0cd2433c61d464666cbae19da16ac197d7a75`.
- Local TCP networking for three party processes when executing MPC.

Existing NumPy installations can run the smoke pipeline directly:

```bash
cd /Users/erdos/Documents/GitHub/sparce-PCA
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m unittest discover -s tests -v
python3 -m sparse_pca.cli validate --config configs/smoke.json --out results/smoke
python3 -m sparse_pca.cli compile-suite --config configs/smoke.json --out results/smoke --mpspdz /tmp/MP-SPDZ
```

For a new Python environment, install the project with `python3 -m pip install -e .`.
The `biology` extra supplies Scanpy and SciPy; the `plots` extra supplies Matplotlib.
The compiler integration test is opt-in:

```bash
SPCA_TEST_COMPILE=1 python3 -m unittest discover -s tests -p test_mpc_compile.py -v
```

## Run the compiled smoke suite in your Terminal

After compilation, run:

```bash
cd /Users/erdos/Documents/GitHub/sparce-PCA
bash scripts/run_smoke.sh
```

If Terminal's `python3` lacks NumPy, select the Python used for compilation:

```bash
SPCA_PYTHON=/Users/erdos/miniconda3/bin/python3 bash scripts/run_smoke.sh
```

The suite runs six jobs sequentially: three representations for a validation-only
operator evaluation, then the same representations for 20 PC1 iterations. Three
party processes run concurrently within each job. Ports start at 19000; keep
other experiment suites from using that port concurrently. To choose a port:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m sparse_pca.cli run-suite --out results/smoke --mpspdz /tmp/MP-SPDZ --port 21000
python3 -m sparse_pca.cli report --out results/smoke
python3 -m sparse_pca.cli bundle --out results/smoke
```

Return **`results/smoke/return-logs.tar.gz`** for verification and debugging. It
contains job metadata, compiler logs, all three party logs for each job, execution
statuses, verification results, and summary tables. Raw input files, private
keys, certificates, and runtime directories are excluded. The smoke fixture is
synthetic; final PCA outputs are present in its logs.

Useful files to inspect individually:

- `results/smoke/summary.csv`: one row per job, correctness and measured costs.
- `results/smoke/<variant>-<task>/compile.log`: compiler output and resource counts.
- `results/smoke/<variant>-<task>/party-{0,1,2}.log`: runtime output and errors.
- `results/smoke/<variant>-<task>/execution.json`: elapsed wall time and exit codes.
- `results/smoke/<variant>-<task>/verification.json`: accuracy checks and status.

The wrapper still bundles logs if execution or verification fails. Before a run,
`report` records `awaiting_execution`; missing execution never counts as a pass.
Each job has an isolated runtime directory, so input files in the MP-SPDZ checkout
are not overwritten. SSL keys are generated locally for that job.
Party certificates contain explicit `DNS:P0`, `DNS:P1`, and `DNS:P2` Subject
Alternative Names for compatibility with OpenSSL 4. The runner checks hostname
verification before startup and automatically replaces legacy CN-only certificates.
The runner links the checkout's shared libraries into that directory so macOS
can resolve MP-SPDZ's relative `libSPDZ.so` dependency. Re-running the smoke
wrapper applies this fix to existing jobs without recompilation.

## Privacy and leakage contract

Three hospitals are also the three computing parties. Use semi-honest replicated
ring sharing with at most one corrupted party. The implementation assumes honest
input preparation and a consistent public gene order. It does not validate
maliciously formed inputs or provide differential privacy for released outputs.

| Representation | Declared public information |
|---|---|
| Dense | Dimensions, gene order, iteration count, seed, numeric configuration |
| Public-support sparse | The above plus padded coordinates and capacity |
| Private-support sparse | Dimensions, gene order, iteration count, seed, numeric configuration, fixed capacities |

All variants secret-share expression values and local column sums. Global means,
intermediate products, norms, and iterates remain secret. The public initialization
vector becomes secret before the first multiplication, giving all timed operator
calls the same secret-vector interface. PC1 jobs reveal only final loadings and
the covariance eigenvalue to the parties. Operator jobs reveal their result solely
for validation and are explicitly marked in their configuration.

Local input manifests contain actual nonzero counts and input hashes for the
experiment coordinator. These are research metadata, not messages released by
the private-support MPC protocol. Preparation runs on public benchmark fixtures;
production hospital input provisioning would require separate owner-side tooling.

## Private-support algorithm

`src/sparse_pca/oblivious.py` contains the exact network imported by the MPC
compiler and exercised with clear scalars in local correctness tests. Inputs
are fixed-capacity secret `(destination, source, value)` tuples. Zero-valued
padding uses valid coordinate zero. This is equivalent to a masked absent entry
for the linear operator; it needs no public occupancy indicator.

For each multiplication, the implementation:

1. Obliviously sorts vector and entry records by source coordinate.
2. Propagates the matching secret vector value and forms entry products.
3. Obliviously sorts products with output markers by destination.
4. Performs a secret segmented sum.
5. Obliviously sorts markers to return a dense secret vector in public order.

Sorting uses a bitonic network whose comparison schedule and memory accesses
depend on public capacity. The output-marker sort prevents secret destination
indices from becoming public array accesses. Repeated coordinates are added.
Transpose multiplication swaps source and destination coordinates.

This is an initial, auditable-by-inspection implementation. It rebuilds routing
every multiplication and uses sequential propagation and segmented sums. Its
communication depth and compile size can be substantial. It is not a formal
security proof or a claim of state-of-the-art sparse MPC performance. A negative
crossover result describes this implementation and its declared capacities.

The public-support diagnostic traverses padded entries with public runtime
indices. It is capacity-based, so it does not represent a tight unpadded CSR
implementation. Dense dot products use MP-SPDZ's fixed-point dot-product primitive.
Public-index accumulation is enclosed in an MP-SPDZ memory-protection scope:
multiple entries and padding can reference the same destination, so preserving
read/modify/write order is necessary for correctness under compiler optimization.

## Numerical correctness

Smoke settings use 20 fractional bits, a 40-bit fixed-point representation, and
a 128-bit ring. Larger configurations use a 60-bit representation and 256-bit ring.
The exporter checks conservative unit-vector product/accumulator bounds, and the
compiler performs additional primitive-specific width checks. These bounds do
not establish stability of every normalization; compare actual outputs to the
reference after execution. Synthetic and PBMC configurations apply declared public
power-of-two input scales of 1/16 and 1/64. Eigenvalues and validation operator
outputs are restored to original units inside MPC. Input scaling is identical
across representations and preserves PC directions; it can affect fixed-point
accuracy. Overflow is rejected rather than silently adjusting these scales.

A public squared-norm threshold must be representable at the configured
precision. The program securely selects a safe denominator and retains the old
vector for a numerically zero norm. Iteration count remains public and fixed.
MP-SPDZ's default probabilistic fixed-point truncation is retained. The Python
quantization model uses deterministic rounding and is an approximation for
calibration, not a bit-for-bit emulator.

Initial acceptance thresholds:

- Operator relative error below `1e-3` (absolute normalization floor `1e-12`).
- PC1 absolute alignment above `0.999` with matched-iteration reference.
- Alignment above `0.999` with direct SVD for nonzero leading eigenvalues.
- Relative eigenvalue error below `1%` and unit-norm error below `0.01`.

Residuals are recorded. A small eigengap can prevent a fixed iteration budget
from meeting the SVD target; that is reported as a failed accuracy target.

## Benchmarking and data

### Controlled sparsity pilot

`configs/sparsity-pilot.json` fixes 24 cells, 12 genes, three equally sized
hospital partitions, uniform support, seed 17, and the numerical settings.
It requests densities 1%, 3%, 10%, 25%, 50%, and 100% (sparsities 99%, 97%,
90%, 75%, 50%, and 0%). Integer nonzero counts slightly change realized levels;
all reports use realized density. No nonzero value threshold is applied.

The pilot measures the centered operator, including global mean preparation,
independently of power-iteration normalization. All input vectors are secret.
Each case uses identical input matrices and vectors across implementations.
It runs one warm-up plus five measured trials per variant: 108 runs total.
Variants are interleaved in seeded randomized order within each repetition
round to reduce machine-temperature and execution-order bias.
Runtime startup is included in wall time and excluded from forward/transpose
kernel timers. Both measures are summarized separately, alongside communication
and accuracy failures. Synthetic capacity per hospital is the declared total
nonzero bound, capped by the public maximum hospital matrix size. At higher
densities padding can occupy the entire matrix; sorting rounds up record counts
to powers of two, so performance can change in steps.

Compile without starting parties:

```bash
PYTHONPATH=src python3 -m sparse_pca.cli compile-sweep --config configs/sparsity-pilot.json --out results/sparsity-pilot --mpspdz /tmp/MP-SPDZ
```

Run the prepared jobs in your Terminal:

```bash
SPCA_PYTHON=/Users/erdos/miniconda3/bin/python3 bash scripts/run_sparsity_pilot.sh
```

The wrapper creates `results/sparsity-pilot/sparsity-summary.csv` and
`return-logs.tar.gz`. A stopped sweep can be resumed with the same command;
completed executions, including failed ones, are retained. To retry failed jobs,
preserve their evidence and prepare a new experiment directory. `run-sweep`
uses no network shaping and serializes cases to avoid port conflicts.
This pilot uses one matrix seed. Any apparent crossover requires confirmation
with additional seeds, larger shapes, and complete PC1 runs. A crossover is
specific to protocol, network, precision, capacity, and representation leakage.

Run a measured local-network benchmark with one warm-up and five repetitions:

```bash
python3 -m sparse_pca.cli benchmark --config configs/smoke.json --out results/benchmark-smoke --task pc1
```

Use a fresh output directory for each benchmark. Compilation is outside trial
timing. Each trial includes startup, inputs, means, all iterations, final
Rayleigh-quotient evaluation, and output. The Rayleigh quotient adds one operator
call beyond the configured iteration count. Timers 3 and 4 therefore accumulate
`iterations + 1` applications in PC1 jobs.

| Timer | Work |
|---|---|
| 1 | Input sharing, including public-support coordinate disclosure |
| 2 | Mean aggregation |
| 3 | Forward product and associated sparse routing |
| 4 | Transpose product and associated sparse routing |
| 5 | Centering correction |
| 6 | Normalization |
| 7 | Final output |

Wall time and each timer are distinct measurements. Party log communication is
summed only if every party reports its sent data. `/usr/bin/time` records peak
memory per party when available; the report gives the maximum party RSS, not
simultaneous aggregate memory. Verbose runtime logs retain additional protocol
statistics; unsupported offline/online counters remain unreported. No fake
preprocessing is used. Local tests and compiler checks produce no runtime claims.

`configs/synthetic.json` includes the proposed shape/density/support/partition
grid. `sweep` runs that grid and preserves case failures. The initial network is
expensive; establish feasibility with smoke and selected small cases first.
Synthetic sweep capacity equals the declared total nonzero count, providing a
fixed upper bound for any party independent of realized row allocation. Capacity
and resulting padding remain part of every comparison.

```bash
python3 -m sparse_pca.cli sweep --config configs/synthetic.json --out results/synthetic-grid --task operator
python3 -m sparse_pca.cli plot --out results/synthetic-grid
```

Summary JSON contains median, min/max, standard deviation, and matched dense
speedups from passing trials. Failed trials remain in `measurements.json`; they
are excluded from speedup estimates. Do not interpret sparse and dense speedups
as comparisons at equal accuracy unless both pass their accuracy checks. Report
sample sizes and failures alongside medians. Plots group shape, support, and
partition separately. WAN shaping and multi-host measurements are future work.

For PBMC3k, install the `biology` extra, then prepare with `configs/pbmc3k.json`.
Filtering requires 200 expressed genes per cell and three cells per gene,
followed by normalization to 10,000 counts and `log1p`. Genes are selected in stable
identifier order; the chosen identifiers are saved. Preparation rejects cells
with zero expression after gene selection. No feature scaling or explicit
centering is applied. PBMC3k is a single healthy-donor workload and supports
systems measurements, not multi-patient or rare-disease conclusions.

## Project layout

- `src/sparse_pca/`: data, references, numerical checks, routing, compiler/runner,
  verification, reporting, plotting, and CLI.
- `mpc/common.mpc`: one parameterized secure program for all six job types.
- `configs/`: smoke, synthetic grid, and PBMC3k settings.
- `tests/`: local tests and opt-in MP-SPDZ compiler integration.
- `scripts/`: installation helper and Terminal smoke runner.
- `results/`: ignored generated inputs, compiled snapshots, logs, and reports.
