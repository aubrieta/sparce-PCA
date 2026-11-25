# Optimized public-support kernel experiment: final three-seed results

## Final conclusions across seeds 17, 7, and 29

The requested experiment is complete at 24 cells x 12 genes. The combined
machine-readable dataset is `results/public-kernel-final/{summary,comparisons}.csv`
and `.json`, with provenance and counts in `audit.json`. It includes only the
corrected-clock seed-17 run and the seed-7/29 confirmations. The initial polling
clock run is excluded from combined inference. Confidence intervals remain paired
within fixture and seed; results are not pooled into an artificial larger sample.

All **1,944 trial executions** passed (1,620 measured and 324 warm-ups).
All 112 validation jobs and all 12 separate PC1 confirmations passed under
rounding-v2. PC1 minimum alignment was 0.99934139 and maximum eigenvalue relative
error was 0.00019319 (0.0193%), within the stated targets. Seed 29 converged more
slowly: its residual was approximately 0.0051 after 20 iterations. Meeting the
alignment/eigenvalue target does not imply identical convergence across seeds.

For the 32-independent-application workload, **dense / sparse-dot matrix-kernel
speedups** (higher is better):

| Zero entries | Seed 17 | Seed 7 | Seed 29 | Predeclared kernel benefit in all seeds? |
|---:|---:|---:|---:|---|
| 98.96% | 2.557x | 1.974x | 1.896x | Yes |
| 96.88% | 1.649x | 2.097x | 1.744x | Yes |
| 89.93% | 1.459x | 1.387x | 1.453x | Yes |
| 75.00% | 1.216x | 1.225x | 1.175x | Yes |
| 50.00% | 0.999x | 1.009x | 0.992x | No |
| 0.00% | 0.960x | 1.005x | 1.005x | No |

At the four highest sparsities, every seed has a median runtime reduction of at
least 10% with the paired 95% CI supporting a speedup. At 50% and 0% zeros no seed
meets that criterion. The observed benefit boundary lies between 50% and 75%
zeros on this coarse grid; it is not a precise or general crossover threshold.
Confidence intervals for every fixture, including negative/null results, are
retained in the combined comparisons file. The widest 75%-zero confirmation
interval is seed 29: 1.175 [1.012, 1.216].

The three predeclared outcomes are distinct:

1. **Implementation improvement:** delayed truncation is effective. Dot sparse
   convincingly beats scalar sparse kernels at density 3.125% and above for all
   three seeds, reaching approximately 5.7x. At density 1.042%, scalar is slightly
   faster for seed 7 and about 6% faster for seed 29; dot is not universally the
   best sparse choice. Grouping alone never meets the 10% improvement criterion
   for batch kernels or complete execution across the 18 fixtures.
2. **Runtime sparsity benefit:** dot sparse has a cross-seed matrix-kernel benefit
   at 75% or more zeros. **No tested fixture meets the >=10% complete-execution
   reduction criterion against dense**, for either workload. Some complete-time
   intervals support smaller improvements. The best dot-versus-dense batch point estimate is
   seed 7 at 98.96% zeros: speedup 1.098 [1.059, 1.136], an 8.9% time reduction.
3. **Communication benefit:** dot sparse reduces total batch communication by
   about 64–65%, 50–54%, and 24–30% at the three highest sparsities. At 75% zeros,
   savings vary from approximately 0.12% to 5.9%; at 50% and 0% zeros there is no
   saving. Scalar and grouped communicate equal totals.

### Accuracy-policy audit

The uniform revised floor is 0.002, with relative term 0.0001. Original failures
remain preserved. Importantly, all 4,608 original/quantized reference checks on
the primary validation vectors also pass the **original** tolerance, as do all
2,160 single-trial output checks and all 3,888 aggregate checksum checks. The only
original-policy failures in the validation audit are the original and quantized
comparisons for the same cancellation vector in the adversarial fixture. This
distinguishes the edge-case tolerance issue from the primary performance findings.
Checksums do not establish per-vector accuracy on every batch trial; independent
per-vector validation is the preceding gate, as specified in the methodology.

### Scope and reproducibility

All conclusions are limited to uniform synthetic support, 24x12 shape, 8/8/8
hospital rows, public coordinates and padding, three-party semi-honest replicated
sharing on localhost, scale 0.25, f=20/k=48/ring=128, and capacity
`min(96,max(1,m))` per hospital. The protocol, support loading, values, vectors,
and schedules are matched across the four new variants. The legacy diagnostic
remains separately labeled. No claim is made for private support, scRNA-seq data,
larger shapes, or WAN networks. Full padding explains the dense-equivalent plateau
at higher density; these results do not isolate an unpadded CSR implementation.

Rebuild the combined tables and audit from the preserved execution logs:

```bash
PYTHONPATH=src /Users/erdos/miniconda3/bin/python3 -m sparse_pca.public_kernel combine --out results/public-kernel-final
```

The sections below preserve the seed-17 findings and wall-clock investigation.

## Corrected-clock run: current conclusions

`results/public-kernel-clock-v2` completed all 648 trial executions successfully,
including 540 measured trials. Every execution records the concurrent blocking
waitpid clock. The pre-existing validation and PC1 confirmations were reused
with provenance; the 648 timing trials are new executions. The earlier run
below is retained for comparison, with its wall-time limitation identified.

For 32 independent applications, medians in milliseconds:

| Zero entries | Dense matrix kernels | Dot matrix kernels | Dense / dot kernel speedup, 95% CI | Dense complete | Dot complete |
|---:|---:|---:|---:|---:|---:|
| 98.96% | 19.812 | 7.748 | 2.557 [2.145, 3.075] | 105.634 | 103.615 |
| 96.88% | 19.619 | 11.896 | 1.649 [1.588, 1.739] | 108.102 | 99.727 |
| 89.93% | 19.630 | 13.455 | 1.459 [1.437, 1.539] | 103.609 | 99.548 |
| 75.00% | 19.848 | 16.326 | 1.216 [1.186, 1.309] | 105.714 | 108.665 |
| 50.00% | 19.825 | 19.842 | 0.999 [0.969, 1.026] | 104.173 | 105.811 |
| 0.00% | 19.666 | 20.483 | 0.960 [0.883, 1.002] | 106.714 | 109.574 |

Dot beats dense under the predeclared matrix-kernel criterion at the four highest
sparsities. No fixture meets the >=10% complete-runtime reduction criterion
against dense. At 96.88% zeros the complete-time ratio is 1.084 [1.029, 1.224]:
there is evidence of some improvement, but its median reduction is only 7.75%,
below the predeclared practical-effect threshold. Other complete-time intervals
include 1. Kernel improvement alone must not be presented as a complete-execution
speedup. Setup, input/mean, correction, and output costs dilute the kernel savings.

Against scalar sparse, dot gives convincing batch-kernel improvements at all six
densities (1.30x–5.68x). It gives convincing complete-time improvement at 10.07%,
25%, 50%, and 100% realized density: respectively 1.276x, 1.675x, 1.902x, 1.989x,
with 95% lower bounds above 1 and >=10% median reductions. Grouping alone meets
neither the kernel nor complete-time practical-effect criterion at any density.
Protocol communication totals match the first run: dot saves about 64%, 50%, and
29% against dense at the three highest sparsities, with negligible/no savings
at the remaining levels.

These are seed-17 results under the rounding-v2 policy and all controls described
below. Confirmation seeds 7 and 29 are the next stage; do not yet claim a
cross-seed crossover or an advantage for larger matrices or hidden support.

Source: `results/public-kernel-optimized/summary.json` and `comparisons.json`.
All 648 trial executions passed under the documented rounding-v2 policy:
2 warm-ups + 10 measured rounds, 6 density fixtures, single and 32-application
workloads, plus the separate single-application legacy diagnostic. This gives
540 measured runs. All four separate 20-iteration PC1 confirmations passed;
alignment exceeded 0.99999999997 and eigenvalue relative error was 1.56e-6.

The original accuracy policy passed 43/44 pre-measurement validation jobs.
Rounding-v2 uses a uniform absolute floor of 0.002 with relative tolerance
0.0001; it passes the same 44 jobs without rerunning them. Original validation
failures and the reason for revising the floor remain recorded. These conclusions
are conditional on that policy and do not assert that all adversarial cases
meet the original 0.001 absolute floor.

## Valid kernel results

Medians below are MP-SPDZ forward-plus-transpose timer totals for **32 independent
applications**, in milliseconds. They exclude centering correction, loading,
startup, and output. Paired 95% bootstrap intervals use ten matching rounds.

| Zero entries | Dense | Scalar sparse | Grouped sparse | Dot sparse | Dense / dot, 95% CI |
|---:|---:|---:|---:|---:|---:|
| 98.96% | 19.260 | 9.627 | 9.546 | 7.236 | 2.662 [2.576, 2.783] |
| 96.88% | 19.102 | 15.513 | 15.696 | 10.960 | 1.743 [1.705, 1.773] |
| 89.93% | 19.214 | 36.999 | 36.301 | 13.201 | 1.455 [1.416, 1.535] |
| 75.00% | 19.245 | 83.320 | 82.235 | 15.998 | 1.203 [1.189, 1.218] |
| 50.00% | 19.233 | 106.959 | 106.419 | 19.366 | 0.993 [0.977, 1.012] |
| 0.00% | 19.206 | 106.430 | 106.313 | 19.254 | 0.998 [0.985, 1.007] |

Dot accumulation meets the predeclared >=10% runtime-reduction criterion against
dense for the first four batch-kernel fixtures. Its matrix-kernel advantage is
not significant at 50% or 0% zeros. Compared with scalar sparse, dot sparse gives
1.33x–5.53x batch-kernel speedups, meeting the criterion at every tested density.
Grouping alone yields roughly 0.99x–1.02x and never meets the 10% criterion.
Compiler evidence confirms unchanged truncation work for grouping and fewer
reductions for dot accumulation, rather than relying on time alone.

This is a one-seed observation at 24 cells x 12 genes, 8/8/8 hospital rows,
public support, scale 0.25, f=20/k=48/ring=128, three-party replicated sharing on
localhost, and capacity `min(96,max(1,m))` per hospital. The benefit is observed
at 75% zeros and absent at 50%; additional seeds and finer sampling are needed
before identifying a crossover. Padding saturates to 288 total slots at 50%
nonzero density, explaining the plateau with dense-equivalent arithmetic.

## Communication

Median total communication for the 32-application workload, in decimal MB:

| Zero entries | Dense | Scalar/grouped | Dot |
|---:|---:|---:|---:|
| 98.96% | 8.438 | 3.694 | 3.055 |
| 96.88% | 8.438 | 7.933 | 4.241 |
| 89.93% | 8.438 | 20.843 | 5.953 |
| 75.00% | 8.438 | 49.129 | 8.428 |
| 50.00% | 8.438 | 65.006 | 8.438 |
| 0.00% | 8.438 | 65.006 | 8.438 |

Dot sparse saves about 64%, 50%, and 29% total communication at the three highest
sparsities. The difference at 75% zeros is only about 0.12%; it is negligible in
practical terms despite deterministic protocol counts. Grouping alone changes
no communication totals. These numbers include protocol/input/output costs;
kernel-only communication is separately available in the machine-readable files.

## Wall-time measurement correction

The original parent runner called `Popen.wait(timeout=...)` on POSIX. Python's
implementation polls with exponential sleep intervals up to 50 ms. Observed
wall times cluster around 87, 142, 197, and 252 ms. This adds detection latency
and can hide or exaggerate complete-execution differences at this matrix size.
**Do not interpret the original complete/cold-workflow speedup intervals as
evidence.** MP-SPDZ internal timers and protocol communication are unaffected.
The old raw data and summaries remain untouched as evidence of the issue.

The runner now supports concurrent blocking `waitpid` calls, deadline enforcement
via a condition variable, and per-party observed completion timestamps. It records
the method and runner source hash. Tests cover all three completions, real local
processes, and deadline cleanup without opening sockets. Existing legacy runner
behavior remains the default for older experiment scripts; the optimized
experiment explicitly opts into the corrected clock.

A separate `results/public-kernel-clock-v2` experiment reuses the immutable
compiled jobs and copies verified correctness/PC1 logs with provenance. It copies
**no timing trials**. All 648 performance trial executions must be repeated in
the user's Terminal. No recompilation is needed. The summary code rejects mixed
clock methods and will not endorse wall-speedup outcomes from the old clock.

```bash
SPCA_OUT=results/public-kernel-clock-v2 \
SPCA_ACCURACY_POLICY=configs/public-kernel-accuracy-v2.json \
SPCA_PYTHON=/Users/erdos/miniconda3/bin/python3 \
bash scripts/run_public_kernel.sh
```

After those wall measurements are verified, run confirmation seeds 7 and 29
using the commands in `PUBLIC_KERNEL_EXPERIMENT.md`, selecting rounding-v2 in
both stages. Until then, findings above establish a seed-17 kernel benefit and
communication benefit; a convincing complete-execution benefit is pending.
