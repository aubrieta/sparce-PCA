# Seed-17 optimized public-support results

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
