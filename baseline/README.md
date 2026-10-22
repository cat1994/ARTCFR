# Comparison algorithms

`run_baselines.py` runs the six comparison curves in Figures 2 and 3:
RM+/CFR+ with quadratic averaging, predictive RM+/CFR+ with last and
quadratically averaged strategies, OMWU/DOMWU, Reg-OMWU/Reg-DOMWU, and R-NaD.
The eight `domwu_weights` profiles run the four Figure 4 entropy weights on
Kuhn and Leduc poker. Each profile uses the corresponding figure's iteration
budget by default.

```sh
python run_baselines.py --list
python run_baselines.py --profile matrix_seed0_omwu --output-root results/omwu_seed0
python run_baselines.py --profile kuhn3_predictive_rm_plus_last --output-root results/pcfr_kuhn
python run_baselines.py --profile leduc3_domwu_weights_kroer17 --output-root results/domwu_kroer17
```

The profiles transcribe the printed values in Tables 2 and 3. `prox_weight`
is the coefficient passed directly to the entropy prox, following the
historical implementation's use of the table's eta parameter. It is not
inverted by the runner. The Figure 4 profiles use the Table 3 coefficient
unless overridden or retuned. To reproduce the published plots from their
saved samples, use `plotting/plot_paper.py`; a rerun with rounded table
coefficients need not match every archived floating-point value.

The comparison loops are extracted from the authors' experiment source at
revision `8b8b45d`, with only the modules needed here. The shared game and
prox interfaces accept both simplex and treeplex domains. The code retains
alternating regret/optimistic updates and the simultaneous R-NaD updates in
that source, its entropy-prox initialization, its initial averaging weight
of one, and the factor-two regularizer decay in Reg-OMWU. Run metadata
records these choices. The R-NaD refresh guard additionally handles an
explicit interval-one override without stalling; the paper profiles use
intervals of 10, 20, or 30 and retain the original schedule.

For a new short parameter search, use 20 log-spaced prox coefficients from
0.01 to 10, or supply an explicit grid. The default tuning horizon is one
tenth of the requested run budget (at least one iteration); use
`--tuning-iterations` to override it:

```sh
python run_baselines.py --profile kuhn3_omwu --tune --tuning-iterations 100 --output-root results/tuned_omwu
python run_baselines.py --profile matrix_seed0_rnad --tune --rt-weight-grid 0.5,0.1,0.05 --rt-step-grid 10,20,30 --output-root results/tuned_rnad
```

Each candidate starts from a fresh solver. Selection uses exploitability at
the specified tuning horizon; the selected solver is then restarted for
the requested full run. `run.json` records every candidate and the resolved
parameters; `trace.csv` contains iteration, exploitability (the sum of both
players' best-response gains), and elapsed seconds. Existing output
directories are never overwritten.
