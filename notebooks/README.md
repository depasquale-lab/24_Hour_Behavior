# Notebooks

Pluto notebooks for the analyses and figures in the paper. The model fits
(GLM-HMM and DDM-HMM posteriors) are produced by the headless fitting scripts
in this folder; everything else — diagnostics, statistics, figure
panels, runs interactively in Pluto.

## 1. Install Julia

These notebooks were developed against **Julia 1.11.7**. Any 1.11.x point
release should work. The easiest cross-platform installer is
[`juliaup`](https://github.com/JuliaLang/juliaup):

```bash
# macOS / Linux
curl -fsSL https://install.julialang.org | sh

# Windows (PowerShell)
winget install julia -s msstore
```

Then pin the channel used in this repo:

```bash
juliaup add 1.11
juliaup default 1.11
```

## 2. Install Pluto

Pluto is listed as a dependency in this folder's `Project.toml`, so once you
instantiate the environment (next section) it is already available. You can
also install it into your global environment if you prefer:

```julia
using Pkg
Pkg.add("Pluto")
```

## 3. Set up the environment

The `notebooks/` folder has its own Julia environment pinned via
`Project.toml` + `Manifest.toml`. From the **repository root**:

```bash
julia --project=notebooks -e 'using Pkg; Pkg.instantiate()'
```

This will resolve and download every package at the exact versions in
`Manifest.toml`. The first run takes several minutes (and triggers
precompilation on first use of each notebook).

Note that `Project.toml` references two path-based dependencies:

- `BehaviorModels` — the package at the root of this repository (resolves
  automatically via the `[sources]` table).
- `DriftDiffusionModels` — expected at
  `\Users\senne\Documents\GitHub\DriftDiffusionModels.jl` on the original
  author's machine. On any other machine, either clone
  [DriftDiffusionModels.jl](https://github.com/depasquale-lab/DriftDiffusionModels.jl) into a sibling folder and
  edit the `[sources]` path in `Project.toml`, or `Pkg.develop` it locally
  before instantiating.

## 4. Launch a Pluto notebook

From the repo root:

```bash
julia --project=notebooks -e 'using Pluto; Pluto.run()'
```

This starts the Pluto server and opens your browser. In the file picker,
navigate to `notebooks/` and open the `.jl` file you want, Pluto will
activate the notebook's environment automatically.

To open one directly without the file picker:

```bash
julia --project=notebooks -e 'using Pluto; Pluto.run(notebook="notebooks/PopulationDDMHMM.jl")'
```

## 5. Notebook map

| Notebook | Purpose |
|----------|---------|
| `DataPreprocess.jl` | Build the cleaned trial-level dataset used by everything downstream. |
| `FitDDMHMMs.jl` | Headless DDM-HMM fitter (also driven by `ddmhmms.sh` on the cluster). |
| `CrossValidationLoop.jl` | Cross-validated state-count selection. |
| `PlotConstrainedBIC.jl` | BIC summaries / state-count selection figure. |
| `GLM_HMM.jl` / `HMM_DDM.jl` | GLM-HMM fits and the DDM-HMM head-to-head setup. |
| `PopulationDDMHMM.jl` | Population-level analyses, parameter contrasts, RM-ANOVA, post-hocs. |
| `PosthocDDMHMM.jl` | Per-rat post-hoc analyses of fitted DDM-HMMs. |
| `ModelCompFigure.jl` | GLM-HMM vs DDM-HMM alignment, NMI permutation tests, BH-FDR. |
| `MiscFigures.jl` | Miscellaneous supplementary panels. |
| `Rat_Daily.jl` | Per-rat / daily exploratory views. |
| `eDDM.jl`, `eDDM_posthoc.jl` | Extended-DDM variants and follow-up. |

## 6. Reproducing the published figures

Each panel below is produced by the listed script. Schematic panels (chamber
diagrams, task timelines, state-transition cartoons) were drawn in BioRender or
Illustrator and have no code counterpart.

### Main figures

| Figure | Panels | Produced by | Output files |
|---|---|---|---|
| **Fig. 1** — Live-in operant facility and task | A–C | *schematics, no code* | — |
| | D | `MiscFigures.jl` §2 | `rat_trial_time_histogram.svg` |
| | E | `MiscFigures.jl` §3 | `rat_daily_acc.svg` |
| **Fig. 2** — Individual variability and diurnal modulation | A–C | `R/GAMS_Flashes_24Hour.R` | `ranked_acc.svg`, `ranked_rt.svg`, `ranked_trials.svg` |
| | D–F | `R/GAMS_Flashes_24Hour.R` | `flashes_acc_vs_tod_marginal.pdf`, `flashes_rt_vs_tod_marginal.pdf`, `flashes_trials_rate_marginal.pdf` |
| **Fig. 3** — Accurate account of behavior | A (cross-validation) | `CrossValidationLoop.jl` | writes `results/`; plotted inline |
| | B–D (RT fit, Q–Q, accuracy) | `PopulationDDMHMM.jl` §12 | `rt_distributions_by_rat.svg`, `qq_population_rts.svg`, `acc_pred_population_rankcolored.svg` |
| **Fig. 4** — Fit for an individual rat | A–G | `PosthocDDMHMM.jl`, with `rat = "Remy"` | `posthoc_remy_*.svg` |
| **Fig. 5** — States across the 24-hour cycle | all | `PopulationDDMHMM.jl` §8–§10 | `population_state_occupancy.svg`, `population_enrichment.svg`, `parameter_boxswarm_by_accRank_population.svg` |
| **Fig. 6** — Trial-to-trial RT correlations | A (parameter ACFs) | `eDDM_posthoc.jl` §4 | `eddm_trial_param_acf.svg` |
| | B (RT ACF vs. both models) | `eDDM_posthoc.jl` §5–§6 | `eddm_rt_acf.svg`, `eddm_rt_acf_ppc.svg` |
| **Fig. 7** — Short-timescale fluctuations | A, B (RT distributions) | `Rat_Daily.jl` §5, §4 | plotted inline |
| | C (posterior decoding) | `Rat_Daily.jl` §3 | plotted inline |
| | D (population ACF) | `Rat_Daily.jl` §2 | `rat_daily_acf_ppc.svg` |

### Supplementary figures

| Figure | Produced by | Output files |
|---|---|---|
| **SF1** — Training procedure and dataset summary | *schematic + summary table, no code* | — |
| **SF2** — Individual GAM fits, reaction time | `R/GAMS_Flashes_24Hour.R` | `flashes_rt_facet.svg` |
| **SF3** — Individual GAM fits, accuracy | `R/GAMS_Flashes_24Hour.R` | `flashes_acc_facet.svg` |
| **SF4** — Individual GAM fits, trial production | `R/GAMS_Flashes_24Hour.R` | `flashes_trials_facet.svg` |
| **SF5** — RT distributions | `PopulationDDMHMM.jl` §12 | `rt_distributions_by_rat.svg`, `accuracy_fit_by_rat.svg` |
| **SF6** — Example fit for an expert rat | `PosthocDDMHMM.jl`, with `rat = "1062"` | `posthoc_1062_*.svg` |
| **SF7** — DDM parameter regressions | `PopulationDDMHMM.jl` §6–§7 | `beta_forest_population.svg`, `beta_forest_population_rt.svg` |
| **SF8** — BIC model comparison | `PlotConstrainedBIC.jl` | reads `results/bic_summary.csv`, `results/bic_winners.csv`; plotted inline |
| **SF9** — Drift rate under high contrast | *not in this repository* — the contrast-manipulation DDMs were fit with the [`rddm`](https://github.com/gkane26/rddm) R package (QMPE), as described in the paper's Methods | — |
| **SF10** — Parameter learning in a multi-level DDM | `eDDM_posthoc.jl` §2–§3 | `eddm_elbo_history_by_rat.svg`, `eddm_hyperparams_by_rat.svg` |
| **SF11** — DDM-HMM and GLM-HMM find similar dynamics | `ModelCompFigure.jl` §4–§6 | `model_comparison_lift_heatmap.eps`, `group_null_hist.eps`, `zscores_by_animal.eps` |
| **SF12** — GLM-HMM states align with task structure | from the GLM-HMM fits (`GLM_HMM.jl`) | plotted inline |
| **SF13** — GLM-HMM state structure in an expert rat | `ModelCompFigure.jl` §7 | `null_1065.eps`, `null_Draco.eps`, `logp_by_animal.eps` |

### Two figures, one notebook

`PosthocDDMHMM.jl` generates both Figure 4 and Supplementary Figure 6. It is
parameterised by a single variable near the top of the notebook:

```julia
rat = "1062"     # Supplementary Figure 6 (expert rat)
# rat = "Remy"   # Figure 4 (representative rat)
```

Set `rat` and re-run the notebook; outputs are written to `results/` as
`posthoc_<rat>_<panel>.svg`.

### Order of operations

Fitting must precede the figure notebooks. From a clean checkout:

```
DataPreprocess.jl        # raw -> processed table (already committed; rerun only if needed)
FitDDMHMMs.jl            # fits every animal; writes ddmhmms/*.bson   (slow; see ddmhmms.sh)
CrossValidationLoop.jl   # state-count selection; writes results/
   |
   +-- PosthocDDMHMM.jl      -> Fig. 4, SF6
   +-- PopulationDDMHMM.jl   -> Fig. 3B-D, Fig. 5, SF5, SF7
   +-- ModelCompFigure.jl    -> SF11, SF13
   +-- eDDM_posthoc.jl       -> Fig. 6, SF10   (requires eDDM.jl first)
   +-- Rat_Daily.jl          -> Fig. 7
   +-- PlotConstrainedBIC.jl -> SF8
   +-- MiscFigures.jl        -> Fig. 1D, 1E
```

Figure 3A is produced by `CrossValidationLoop.jl` itself. Several notebooks
(`PlotConstrainedBIC.jl`, `GLM_HMM.jl`, `HMM_DDM.jl`, and parts of
`Rat_Daily.jl`) render their panels inline in Pluto rather than writing files;
use Pluto's export or right-click-save to obtain them.

`R/GAMS_Flashes_24Hour.R` is independent of the Julia pipeline and reads
`data/processed_rat_data.csv.gz` directly. It produces Figure 2 and SF2–SF4.

## 7. A note on figures

Figures rendered by these notebooks **will not match the published versions
pixel-for-pixel**. The notebook outputs are the analytical source figures
the published panels were exported as vector graphics (SVG / EPS) and then
post-processed in Adobe Illustrator for typography, legends, panel
composition, color harmonization, and layout. The numbers, statistics, and
geometry are the same; the polish is not. If you have any questions, please email
rsenne at bu dot edu.
