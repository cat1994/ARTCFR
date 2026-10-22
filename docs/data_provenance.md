# Paper figures and historical data

The 122 text files in `data/historical/` are the original recorded data used
in Figures 2-6 of the paper. They produce 130 curves with 13,226 selected
records across 16 panels and five shared legends. Some records are used in
more than one panel. `data/historical/SHA256SUMS` records the unchanged file
hashes; its paths are relative to the release root.

Run `python plotting/plot_paper.py --output results/figures` from the release
root. The script resolves its data and relative output paths from the release
root, so it can also be invoked using its absolute path from another working
directory. It writes 21 vector PDFs and a rendering manifest. NumPy,
Matplotlib and pypdf are required; external TeX and Gnuplot are not needed.
Computer Modern fonts are provided by Matplotlib and embedded in the PDFs.

The plotting manifest records each curve's label, data file, SHA256, line
style and axis limits. It was derived from the original Gnuplot mappings.
The plotter uses column 1 for iterations and column 3 for exploitability.
It preserves all selected rows and their order. Nonfinite and nonpositive
values become gaps on logarithmic axes; there is no smoothing, resampling
or artificial numerical floor. The plotted metric is NashConv, the sum of
the two players' unilateral best-response gains. OpenSpiel's function named
`exploitability` returns half of this sum in a two-player game.

| Paper figure | Panels | Complete iteration budget | Shared legend |
| --- | --- | --- | --- |
| 2 | `matrix_game_seed_0` through `matrix_game_seed_3` | 2,000 each | `legend-matrix-games` |
| 3 | `kuhn-3`, `leduc-3`, `goofspiel-4`, `liarsdice-6` | 400; 10,000; 2,000; 100 | `legend-efg-games` |
| 4 | `kuhn-3-domwu`, `leduc-3-domwu` | 400; 10,000 | `legend-domwu-weights` |
| 5 | `matrix_game_rtdrm_seed_0` through `matrix_game_rtdrm_seed_2` | 2,000 each | `legend-rtdrm-ablation` |
| 6 | `kuhn_3_rtdcfr`, `leduc_3_rtdcfr`, `goofspiel_4_rtdcfr` | 400; 10,000; 2,000 | `legend-rtdcfr-ablation` |

The full file-to-curve mapping is in `plotting/paper_figures.json`. Figures
5 and 6 each use three panels in one row. Their discount pairs are
`(1.5, -inf)`, `(1.5, 0)`, `(1.5, 0.5)`, `(2, -inf)`, `(2, 0)` and
`(2, 0.5)`, followed by the RTRM+ or RTCFR+ comparison curve.

## Historical mapping details

For matrix seed 3 in Figure 2, the saved Gnuplot mapping labels the
`*_rtcfr_plus.txt` file as RTDRM and `*_rtdcfr.txt` as RTRM+. This is a
confirmed label swap: new RTCFR+ and RTDCFR runs each exactly match all
102 saved points of their correspondingly named file over 2,000 iterations;
the cross-matched maximum difference is 0.017221592938594335.
The default command retains the historical mapping to reproduce the paper.

To correct these two curve labels while retaining their algorithm colors,
run:

```sh
python plotting/plot_paper.py --correct-known-labels --output results/figures_corrected
```

This exchanges only the two dataset mappings in the seed 3 panel. Data files
remain unchanged, and all other panels and legends are identical. Corrected
output requires a separate directory so that historical figures are preserved.
The generated `render_manifest.json` records the selected mapping mode and
the correction.

The main RTDCFR curve and the `(2,0)` ablation curve are not universally
the same recorded run. Kuhn and Leduc reuse the main file; matrix seeds
0-2 and Goofspiel use separately named `*_2_0.txt` files. They are retained
as separate inputs. The Goofspiel adaptive RTCFR+ main curve comes from
`goofspiel_4_adp_rtcfr_plus11.txt`.

The historical Liar's Dice data uses the no-wild-face variant with one
six-sided die per player; see [game rules](game_rules.md). Later OpenSpiel
test runs are not included in the historical paper data.

## Plot reproduction and fresh solver runs

Replotting these recorded data reproduces the paper curves. Fresh solver
runs should use the explicit paper configurations and write to a separate
output directory. Historical EFG curves use transformed continuation;
the `local_rt` variant uses local reward transformation and the current
two-threshold adaptive schedule. These are separately named variants.
Some Goofspiel adaptive settings are documented implementation defaults,
rather than recovered settings for the old numbered run. Exact floating
point trajectories can also depend on the recorded numerical environment.

The data was imported from the camera-ready figure archive, whose SHA256
is recorded in `plotting/paper_figures.json`. No solver was rerun to create
or replace the historical data in this release.
