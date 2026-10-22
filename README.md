# Adaptive Reward Transformation: experiment reproduction

Code and experimental data for **Efficient Last-Iterate Convergence in Regret
Minimization via Adaptive Reward Transformation**, by Hang Ren, Yulin Wu,
Shuhan Qi, Jiajia Zhang, Xiaozhen Sun, Tianzi Ma, and Xuan Wang. JMLR accepted
manuscript 24-2008 (2026).

The package provides two workflows:

1. **Rebuild Figures 2–6 from the saved paper data.** The 122 historical data
   files and their recorded curve mappings are included.
2. **Run the algorithms again.** Explicit configurations cover the RT methods,
   comparison baselines, discount ablations, and dilated-weight comparison.
   New runs record their settings and environment in separate output folders.

## Installation

The tested server environment is Python 3.10.12. No GPU, OpenSpiel, external
game files, or native game-solving library is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the commands below from the package directory. Each entry point also
resolves its bundled input files relative to its own location. Requirements
for the separately verified historical Python 3.7 numerical environment are
listed in `requirements-historical.txt`; that file does not install plotting
dependencies.

## Rebuild the paper plots

```bash
python plotting/plot_paper.py --output results/figures
```

This writes 16 vector plot panels and five shared legends. The manifest
`plotting/paper_figures.json` specifies the input file, original columns,
curve label, checksum, and axes for each panel. No numerical result is
recomputed by this command. Figure-to-file mappings and data formats are
documented in [data provenance](docs/data_provenance.md).

The historical Figure 2 Seed 3 mapping interchanges the RTDRM and RTRM+
labels. Numerical replay identifies which data belong to each algorithm.
To draw the corrected labels in a separate directory:

```bash
python plotting/plot_paper.py --correct-known-labels --output results/figures_corrected
```

Both modes use the same original numerical data. The default reproduces the
historical paper mapping; the optional mode corrects this one confirmed
label error.

## Run RT experiments

For all methods in a paper experiment group, use the combined entry point:

```bash
python run_paper.py --experiment main --output-root results/all_main
python run_paper.py --experiment discount-ablation --output-root results/all_discount
python run_paper.py --experiment weights --output-root results/all_weights
```

The groups contain 80 main runs, 42 discount-ablation runs, and eight
dilated-weight runs. Add `--game kuhn` to select one game, `--dry-run` to
inspect the jobs, or `--iterations 2 --num-output 3` for a short execution
check. Full groups use the original figure budgets and can take substantial
time on the larger games.

List the 32 named main-experiment profiles or run one example:

```bash
python run_experiments.py --list-profiles
python run_experiments.py --paper-profile kuhn3_adaptive_rtcfr_plus --iterations 400 --output-root results/kuhn_adaptive
```

The batch commands use the figure-specific iteration budgets in
`configs/rt_suites.json`:

```bash
python run_experiments.py --suite main --output-root results/rt_main
python run_experiments.py --suite discount-ablation --output-root results/discount_ablation
```

Use `--dry-run` to inspect resolved parameters first, `--game kuhn` to select
one game from a suite, or `--iterations 2` for a short execution check. Every
run writes `run.json` and `trace.csv`; batches also write `suite.json`.
Existing output folders are rejected to preserve earlier runs.

The default `--variant paper` follows the recorded paper-experiment
continuation and reference schedules. `--variant local_rt` uses the local RT
continuation and two-threshold reference schedule analyzed in the
camera-ready formulation. The selected variant is recorded in each run.
The parameter provenance in `configs/rt_profiles.json` distinguishes reported
settings, reconstructed curves or prefixes, and implementation defaults.

The discount ablation includes beta = negative infinity, implemented as
negative-regret clipping starting from the first iteration. A single run can
select it using `--dcfr-beta=-inf`.

## Run comparison algorithms

```bash
python run_baselines.py --list
python run_baselines.py --profile matrix_seed0_omwu --iterations 2000 --output-root results/matrix_omwu
```

The 48 main baseline profiles cover RM+/CFR+, predictive RM+/CFR+ with last
and average evaluation, (dilated) OMWU, regularized OMWU, and R-NaD. Eight
additional profiles cover the four dilated weights on Kuhn and Leduc.
See [baseline usage and conventions](baseline/README.md) for
weight settings, update order, and parameter provenance.

## Tune parameters with a short budget

For RT methods, the default search uses one tenth of the full-run iteration
budget. The candidates and selection rule are saved before execution:

```bash
python tune_rt.py --paper-profile kuhn3_adaptive_rtcfr_plus --mu-grid 0.1,0.05,0.01 --T-grid 5,10,20 --output-root results/tune_kuhn --run-selected
```

`--selection final` compares the last-strategy gap at the end of the short
budget. `--selection best` compares the best sampled last-strategy gap within
that budget. `--tuning-iterations 100` sets a specific short budget.
`--run-selected` restarts the chosen configuration for the full budget.
These are new tuning runs; they do not alter historical data or overwrite
the named paper profiles.

The baseline runner offers `--tune`, a default 20-point logarithmic prox
coefficient grid from 0.01 to 10, and explicit regularization-weight and
reference-interval grids. Its resolved selection settings are stored with
the new run.

## Game definitions and validation

[Game rules](docs/game_rules.md) specify the supported configurations and
the exploitability convention. In particular, the paper's Liar's Dice game
has **no wild faces**. Historical figures and reruns use the same Python
game generators.

The implementation checks can be run with:

```bash
python tests/rt_checks.py
```

See [validation](docs/validation.md) for executed checks and their scope.
Saved-curve reconstruction and numerical reruns are separate validations:
software versions, floating-point arithmetic, and rounded reported baseline
parameters can affect a newly generated trajectory.

## Package contents

| Path | Contents |
| --- | --- |
| `sequence_form/` | RT implementation and required game/domain modules |
| `baseline/` | Comparison algorithms and explicit baseline profiles |
| `configs/` | RT profiles, iteration budgets, ablations, and tuning grids |
| `data/historical/` | Original numerical files used in Figures 2–6 |
| `plotting/` | Standalone plotting script and exact curve mappings |
| `tests/` | Small implementation checks |
| `docs/` | Game, data, and validation notes |

Generated outputs go under `results/` and are not part of the source archive.
Author attribution is retained in [ATTRIBUTION.md](ATTRIBUTION.md); citation
metadata are provided in `CITATION.cff`.
