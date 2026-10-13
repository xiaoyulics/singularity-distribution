# The Singularity Has a Distribution — code

Code and numerical outputs for the paper
**"The Singularity Has a Distribution: Exact Timing Laws and Detection Limits for a Stochastic Intelligence Explosion."**

Every number quoted in the paper is computed by these scripts (none is typed by hand), and every
figure is regenerated from them.

## Contents

| Path | Purpose |
|---|---|
| `code/timing.py` | Library: exact law of the explosion time (alternating q-series with certified truncation and automatic working precision), quantiles, exact fixed-sample detection errors (hypoexponential law of the likelihood ratio), illustrative prior, delayed-feedback ODE solver. |
| `code/make_numbers.py` | Computes all quantities used in the paper and writes `generated/numbers.json` and `generated/numbers.tex` (LaTeX macros). |
| `code/run_checks.py` | Regression targets, precision doubling, structural checks, Monte Carlo cross-checks with standard errors, delay-equation grid refinement. Writes `generated/checks_report.txt`; exits with status 1 on any failure. |
| `code/check_small_time.py` | High-precision evaluation of the small-time tail at q = 1/2 (up to 18 000 digits). Writes `generated/small_time.json`. |
| `code/make_figures.py` | Figures 2–6 (matplotlib PGF backend, text set by LaTeX). |
| `code/make_hero.py`, `code/make_hero_labels.py` | Figure 1. |
| `generated/` | Outputs of the scripts above, as used in the paper. |
| `figs/` | The figures, as used in the paper. |

## Reproducing

```sh
pip install -r requirements.txt
sh reproduce.sh          # or, on Windows PowerShell:  ./reproduce.ps1
```

`make_figures.py` and the labelling step of Figure 1 need a LaTeX installation with the
`cochineal`, `newtx` and `plex` packages (they set figure text in the paper's fonts).
Run time: about one minute each for the numbers and the checks; the high-precision small-time
check and the exact detection-threshold grid (`make_figures.py --recompute`) take 10–20 minutes.
Random seeds are fixed in each script.

Tested with Python 3.13, numpy 2.4, scipy 1.17, mpmath 1.3, matplotlib 3.11, pillow.
