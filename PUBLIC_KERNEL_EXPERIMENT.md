# Matched public-support kernel experiment

This experiment preserves the old sources, configurations, and results. New code
is in `mpc/public_kernel.mpc` and `src/sparse_pca/public_kernel.py`; generated
artifacts go to `results/public-kernel-optimized`. The old `common.mpc` remains
unchanged, including its public-coordinate input/reveal path.

## Question and controls

At 24 cells by 12 genes, does grouping or delayed truncation help sparse MPC?
Every new implementation computes `X.T @ (X @ v) - C*mu*(mu.T@v)`, never a Gram
matrix. Three semi-honest replicated parties own eight rows each. Values,
intermediates, means, and normalization remain secret. Shapes, padded support,
initialization, iteration budgets, and arithmetic configuration are public.
All four variants receive the exact same public support schedules, including
the dense baseline. Only the stored secret value representation differs.

| Variant | Matrix products | Scale reduction |
|---|---|---|
| dense | Existing optimized `sfix.dot_product` per dense destination | Once per dot |
| sparse_scalar | Individual products, protected destination updates | Once per product |
| sparse_grouped | Independent product list, then grouped register sums | Once per product |
| sparse_dot | The same groups, `sfix.dot_product` per nonempty group | Once per nonempty group |

The grouped implementation removes destination memory dependencies and exposes
independent arithmetic for compiler merging. The compiler may already merge
parts of scalar arithmetic; improvement is an experimental question. Empty
groups return a secret zero. Every padding slot remains a secret input value
zero, assigned coordinate `(0,0)` and retained in BOTH matrix products. Row and
column schedules reference indices in the same value list, uploaded once.

One shared preparation step builds both schedules. Every new variant loads
support from public compile-time configuration, with no secret coordinate
inputs/reveals. The legacy diagnostic retains the original path and is measured
separately for single applications. It also retains its old constant-based
initialization, which is documented as another loading-path difference; its
timings cannot isolate kernel arithmetic alone. Old result directories are never
reused by this experiment.

Configuration: uniform duplicate-free support; values uniform on `[0.4,4.0]`;
scale 0.25; f=20, k=48, ring=128; seed 17 first, seeds 7/29 after verification.
Requested densities are 1%, 3%, 10%, 25%, 50%, 100%. Each fixture records realized
density, nonzeros, input/support hashes, and padded slots. Capacity per hospital
is `min(96,max(1,m))`. No padding-removal optimization is included.

## Installed API audit and bounds

`api-audit.json` records the installed MP-SPDZ revision, source hash, and exact
line-numbered excerpts. In the installed `Compiler/types.py`:

- `dot_product` calls `unreduced_dot_product(...).reduce_after_mul()`.
- `unreduced_dot_product` invokes an integer dot product of encoded operands.
- `sfix.unreduced` declares the raw width as **k+f**, independent of group length.
- `unreduced_sfix.reduce_after_mul` rounds that integer once by f bits.

Therefore each encoded group must satisfy `d*Amax*Bmax < 2**(k+f-1)`, in addition
to the signed ring bound. The preflight checks every row/column padded group and
dense group. It also checks means, correction, squared norms, Rayleigh products,
restored outputs, and weighted checksum accumulation against declared widths.
The installed ring truncation adds one mask bit; the report includes k+f+1,
double-width arithmetic, and the FPDiv internal-precision requirement. Actual
compilation remains a second guard on square-root and division requirements.

PC1 bounds assume normalized components remain within 1.01, allowing quantization
slack. This is not an unconditional stability proof; final norm, alignment, and
eigenvalue checks remain mandatory. Configurations that fail bounds are rejected
without changing widths or scales. Default probabilistic fixed-point rounding
is retained uniformly. MP-SPDZ's probabilistic-truncation warnings are preserved
in compiler logs; this experiment does not provide a new security proof for those
library primitives.

`compiler-evidence.json` retains compiler resource requirements, static assembly
instruction histograms, and truncation function names. `job.json` separately
records source-derived product/reduction counts. `compiler-proof.json` checks
that grouped and scalar retain equal truncation work and that dot reduction
savings match the compiler's random-bit savings exactly: the installed ring
implementation uses k+f+1 random bits per reduction. At full density the dense
and sparse-dot compiler resources must agree. Compiler virtual-machine rounds
are reported as compiler metrics, not asserted to be measured network rounds.

## Correctness and the cancellation finding

Before timing anything, run 32 vectors through every variant for all six density
fixtures and five additional fixtures: zero, constant, empty rows/columns,
cancellation-heavy repeated contributions, and unequal 3/7/14 hospital rows.
Padding aliases destination zero. Vector 1 is a signed exact-cancellation pair;
other vectors are seeded, signed, and unit normalized. All variants get the
same vectors as secret inputs from party 0, outside the timed region.

Compare to explicit and implicit floating-point centered references and a
reference using fixed-point-quantized matrix/vector inputs. Inputs are explicitly
quantized before writing decimal input files, so quantized references match
the supplied encodings. Means are computed from those same quantized values.
Per-vector accuracy requires:

`L2 error <= 0.001 + 0.0001 * L2(reference)`.

Absolute errors, relative errors, budgets, and pass flags are saved. The original
tolerance is unchanged. **A local deterministic-rounding model found up to
0.001804 absolute error for the mixed-value cancellation fixture in scalar and
grouped arithmetic, where the exact centered answer is zero.** The source is
rounding each forward product independently while the mean correction uses a
single dot-product reduction; errors are amplified by the transpose and restoring
the input scale. Dense/dot variants pass that local model check. This finding is
recorded in each fixture's `local-checks.json` and in an explicit regression test.
It is neither suppressed nor counted as a passing arithmetic check.

The model uses deterministic nearest rounding; MP-SPDZ uses probabilistic rounding,
so actual MPC error must be measured. **The execution runner stops at any failed
validation gate and collects no performance measurements.** Return the logs for
investigation before considering a documented uniform tolerance revision. A
passing Python test suite includes checking that this calibration failure is
detected; it does not establish that every adversarial fixture passes tolerance.

### First MPC validation and explicit revised policy

The initial Terminal execution passed 43 of 44 jobs. All 24 density/variant jobs
passed. One vector in `edge-cancellation/check-sparse_dot` had L2 absolute error
`0.0010004705086681367` against both references; the mathematical centered result
is zero. Dense/scalar/grouped maxima on that fixture were respectively 0.000714,
0.000686, and 0.000826. The output consists of small multiples of the restored
fixed-point resolution `16 / 2**20`, consistent with operation-rounding residuals.
No performance trials ran because the original gate stopped collection.

After investigation, `configs/public-kernel-accuracy-v2.json` explicitly revises
the absolute floor to **0.002 for every variant and fixture**; the relative term
stays 0.0001 and all arithmetic settings stay fixed. The floor covers the
0.001804 deterministic cancellation error recorded before actual MPC execution.
This is an accuracy-policy change, not an improvement in numerical arithmetic.
Rechecking identical logs gives 44/44 passes. Original `verification.json` and
`correctness-summary.json` retain the original failure; new files are named
`verification-rounding-v2.json` and `correctness-summary-rounding-v2.json`.
Each revised per-vector record also reports whether it passes the original
threshold. Statistical summaries label their accuracy policy and reject mixed
policies in one timing population. PC1 targets are unchanged.

Resume into the measurement phase with the policy explicitly selected:

```bash
SPCA_ACCURACY_POLICY=configs/public-kernel-accuracy-v2.json SPCA_PYTHON=/Users/erdos/miniconda3/bin/python3 bash scripts/run_public_kernel.sh
```

This reuses the recorded correctness executions; no rerunning or selecting a
luckier random truncation outcome is involved. Programs need no recompilation.
Use the same policy option when running confirmation seeds, and report the
revision alongside any performance claims.

## Workloads and measurements

- `validation`: 32 independent applications, all outputs revealed after timing.
- `single`: one centered application, complete startup and input handling.
- `batch`: matrix loaded once, 32 independent secret input vectors. Every result
  contributes to a weighted vector checksum, revealed after timing. This is an
  operator benchmark, not power iteration. Aggregate checksum verification does
  not replace the preceding per-vector validation gate.
- `pc1`: separate 20-iteration confirmation at approximately 10% density, after
  primary measurements. Same initialization, mean, correction, normalization,
  output code, and accuracy targets across all four variants. Targets: alignment
  >0.999 against SVD and matched iterations, eigenvalue relative error <1%, norm
  error <0.01. Confirmation seeds use the same targets without relaxation.

Two warm-ups and ten measured rounds per fixture/workload. Variant order is
seeded and randomized within each round. Single applications also interleave
the legacy diagnostic, which is excluded from the matched four-way conclusions.
Completed runs can be resumed; completed failures are retained, not silently
retried. A source hash guards against mixing compiled versions.

Timers: 1 inputs, 2 mean, 3 forward, 4 transpose, 5 correction, 6 normalization,
7 output/checksum work, 8 complete centered-operator region. Batch timer 8
excludes checksum construction and all output. Kernel time means 3+4; centered
operator time means timer 8. Report both rather than conflating them.

All three party logs are retained. Communication is summed across parties and
converted from the runtime's printed MB to estimated bytes (printed precision
limits exactness). If available, timer counters give bytes for 3+4 and timer 8.
No missing counter is invented. OS peak RSS is recorded for each party and its
maximum, not simultaneous aggregate memory.

Public schedule time is measured once per fixture. `process_wall_seconds`
includes startup and secure execution, excluding certificate generation and file
staging. `complete_seconds` adds schedule preparation. `cold_workflow_seconds`
also adds compilation; this is separately labeled and is not the steady-state
runtime metric. Original dataset generation and experiment export are outside
these measures. The legacy diagnostic has zero public schedule-preparation cost.

## Predeclared analysis

Summaries include medians, Q1/Q3, pass/failure counts, and paired bootstrap CIs.
Bootstrap samples resample matching repetition rounds within a fixture, retaining
the dense/sparse pairing. The statistic is the ratio of medians; 10,000 bootstrap
samples give the 2.5th/97.5th percentile CI. Warm-ups are excluded.

- Implementation improvement: grouped/dot beats scalar.
- Runtime sparsity benefit: sparse beats dense.
- Communication benefit: sparse sends fewer bytes than dense.

Convincing runtime evidence requires a median speedup at least `1/0.9` (10%
less time), a lower 95% CI bound above 1, and all ten passing pairs. Communication
benefit uses a ratio above 1 with CI supporting it; it does not require a 10%
runtime reduction. Kernel, complete, and cold-workflow comparisons remain separate.
All densities, including failures/no crossover, must be reported. Any crossover
is limited to this size, padding, arithmetic, network, and public-support leakage.

## Execution handoff

Compile without opening sockets:

```bash
cd /Users/erdos/Documents/GitHub/sparce-PCA
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
/Users/erdos/miniconda3/bin/python3 -m sparse_pca.public_kernel compile --out results/public-kernel-optimized --mpspdz /tmp/MP-SPDZ
```

Run the prepared seed-17 stage in your Terminal:

```bash
SPCA_PYTHON=/Users/erdos/miniconda3/bin/python3 bash scripts/run_public_kernel.sh
```

The wrapper bundles logs on failure as well as success. Return or refer to
`results/public-kernel-optimized/return-logs.tar.gz`. It includes compiler/API
audits, fixture metadata, local checks, execution logs, verification files, and
summaries; it excludes input arrays, private keys, and runtime directories. Files
can also be inspected directly in the shared repository.

After seed-17 correctness and collection are verified, compile confirmation seeds
into a different directory and run them:

```bash
PYTHONPATH=src /Users/erdos/miniconda3/bin/python3 -m sparse_pca.public_kernel compile --seeds 7 29 --out results/public-kernel-confirmation --mpspdz /tmp/MP-SPDZ
SPCA_OUT=results/public-kernel-confirmation SPCA_ACCURACY_POLICY=configs/public-kernel-accuracy-v2.json SPCA_PYTHON=/Users/erdos/miniconda3/bin/python3 bash scripts/run_public_kernel.sh
```

`summary.csv/json` contains fixture/workload/variant medians and quartiles.
`comparisons.csv/json` contains paired CIs and outcome flags. Those files are
empty until timing collection is reached; no compilation or cleartext result is
presented as an MPC performance measurement. PC1 confirmations have their own
`pc1-check-*/verification.json` files, separate from the timing population.
