# Validation

The release preparation checks distinguish preserved input data, algorithm
equivalence after code cleanup, and execution of new experiments.

## Saved paper figures

- All 122 historical input files retain their original SHA-256 checksums.
- The figure mappings select 130 curves and 13,226 data records.
- The 16 generated panels and five legends match the paper figures after
  rendering, pixel for pixel, in the checked local plotting environment.
- The optional known-label correction changes only the Seed 3 panel in
  Figure 2; the other 20 generated assets render identically.

For the Seed 3 correction, the fixed RTRM+ and RTDRM configurations were each
run for 2,000 iterations in Python 3.10.12 / NumPy 1.26.4 / SciPy 1.15.3.
Each matched the 102 saved records in the correspondingly named data file
with maximum absolute gap difference zero. The original plot mapping had
interchanged their labels; the cross-comparison difference was approximately
0.01722. No saved numerical data were changed.

## Algorithm checks after cleanup

- RT code: 16 comparisons against the captured implementation passed with
  exact per-iteration gap agreement. They cover the four RT methods on
  matrix seed 1 (200 iterations) and Kuhn poker (40 iterations), under both
  continuation variants. This check used the historical Windows Python
  3.7.16 / NumPy 1.21.6 / SciPy 1.7.3 environment.
- Six small RT implementation tests cover finite updates, first-iteration
  clipping for beta = negative infinity, continuation variants, and the
  fixed-reference-update special case.
- Baselines: the six comparison configurations on matrix seed 0 and Kuhn
  poker each matched the paper-era implementation for 100 iterations
  (1,212 sampled gaps in total, all absolute differences zero).
- All eight dilated-weight configurations passed short executions. R-NaD
  reference refresh with interval one was checked separately.
- Entry-point checks exercised explicit parameters, negative-infinity
  clipping, a short discount-ablation suite, grid selection and restart,
  and refusal to overwrite an existing run directory.

These checks support the correctness of the extracted execution paths. They
do not assert that every reported curve has been independently rerun to its
full horizon. The saved historical data provide the figure-reconstruction
path; configuration provenance states the available reconstruction evidence
for new numerical runs.

## Standalone execution on 2026-10-22

The extracted package was executed from a separate working directory, with
the original experiment repository absent from the Python import path.
The environment was Python 3.10.12, NumPy 1.26.4, SciPy 1.15.3, six 1.17.0,
Matplotlib 3.7.5, and pypdf 6.10.0.

- All 80 main configurations, 42 discount-ablation configurations, and
  eight dilated-weight configurations completed two iterations each.
- The six RT implementation tests passed in this environment as well.
- RT tuning evaluated a four-candidate grid for two iterations, selected
  by the best sampled gap, then restarted the selected configuration for
  ten iterations. Baseline tuning evaluated two prox coefficients over
  two iterations and restarted the selected configuration for ten.
- The resulting 132 trace files contained 396 finite numerical records.
- Both plotting modes generated all 21 PDFs and checked the 13,226 input
  records. No external TeX installation was needed.

These short executions check the packaged configurations and entry points;
the full-budget historical replay checks are those listed above.
