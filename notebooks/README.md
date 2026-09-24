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

The `notebooks/` folder has its own Julia environment, defined by
`notebooks/Project.toml`. From the **repository root**:

```bash
julia --project=notebooks -e 'using Pkg; Pkg.instantiate()'
```

This resolves and downloads every package. `Manifest.toml` is not committed
and `Project.toml` has no `[compat]` bounds, so you get the latest releases,
which may be newer than the versions used for the paper. The first run takes several minutes, and each notebook
precompiles the first time you open it.

Note that `Project.toml` references two path-based dependencies:

- `BehaviorModels` — the package at the root of this repository (resolves
  automatically via the `[sources]` table).
- `DriftDiffusionModels` — expected at `../../DriftDiffusionModels.jl`
  relative to `notebooks/`, i.e. in the folder that contains this
  repository. Clone
  [DriftDiffusionModels.jl](https://github.com/depasquale-lab/DriftDiffusionModels.jl)
  there, or point the `[sources]` path in `Project.toml` at wherever you
  cloned it.

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
| `FitDDMHMMs.jl` | Headless DDM-HMM fitter (also driven by `cluster/ddmhmms.sh`). |
| `CrossValidationLoop.jl` | Cross-validated state-count selection. |
| `PlotConstrainedBIC.jl` | BIC summaries / state-count selection figure. |
| `GLM_HMM.jl` / `HMM_DDM.jl` | GLM-HMM fits and the DDM-HMM head-to-head setup. |
| `PopulationDDMHMM.jl` | Population-level analyses, parameter contrasts, RM-ANOVA, post-hocs. |
| `PosthocDDMHMM.jl` | Per-rat post-hoc analyses of fitted DDM-HMMs. |
| `ModelCompFigure.jl` | GLM-HMM vs DDM-HMM alignment, NMI permutation tests, BH-FDR. |
| `MiscFigures.jl` | Miscellaneous supplementary panels. |
| `Rat_Daily.jl` | Per-rat / daily exploratory views. |
| `eDDM.jl`, `eDDM_posthoc.jl` | Extended-DDM variants and follow-up. |

### Added in revision

These are plain Julia scripts rather than Pluto notebooks. Run each one from the
repository root with `julia --project=notebooks notebooks/<Script>.jl`. The
slow fits are written as SGE array jobs; the matching `cluster/*.sh` wrappers are
listed in the [cluster scripts](#cluster-scripts) table below.

| Script | Purpose |
|--------|---------|
| `FitConstrainedDDMHMMs.jl` | Constrained (parameter-tied) K = 4 DDM-HMM fitter. It fits each tying config (`full`, `τ`, `a₀`, `τ+a₀`, `v`, `B`) by Baum–Welch and scores it by BIC. It also defines the shared fitting code that the other scripts `include`. |
| `FitOneRat.jl` | Runs `FitConstrainedDDMHMMs.jl` for one rat, as one array task. |
| `FitOneRatGradient.jl` | Same configs as above, but fit by direct L-BFGS on the marginal likelihood instead of Baum–Welch. |
| `DirectGradientDDMHMM.jl` | Single-rat test of the direct-gradient fitter (30 restarts). |
| `MergeBicSummaries.jl` | Combines the per-rat BIC CSVs into `results/final_ddmhmms/bic_summary.csv`. |
| `ConvertConstrainedToDDMHMMFit.jl` | Re-saves the final constrained fits in the `ddmhmms/*_compat.bson` format that the older notebooks read. |
| `ExtractStateParameters.jl` | Reads the final K = 4 fits and writes per-rat, per-state DDM parameters, posterior-weighted behavior, psychometric/chronometric curves and transition matrices. Every state-level figure below reads its output. |
| `PlotStateComparability.jl` | Cross-animal comparability of states: drift against accuracy rank, within-rat Kendall τ, and accuracy against v·B. |
| `PlotStateBehavior.jl` | Behavior conditioned on state: psychometric functions, RT by rank, and psychometric slope by rank. |
| `StateEngagementITI.jl` | Tests whether the states predict trial initiation (ITI, trial rate, breaks). The model never sees these variables. |
| `PlotStateEngagementITI.jl` | Figure for the analysis above. |
| `ParameterRecovery.jl` | Parameter recovery with ground truth matched to each animal. It simulates from each rat's final fit, then refits blind, at full size and at 10/25/50 % of sessions. |
| `PlotParameterRecovery.jl` | Parameter-recovery figure and summary tables. |
| `StateSweepDaily.jl` / `MergeStateSweep.jl` | Sweeps K = 1…5 on the session-based ("daily") animals and records full-data logL, AIC and BIC. |
| `CrossValidateStatesDaily.jl` / `MergeStateSweepCV.jl` | Session-blocked 5-fold cross-validation over K = 1…5 for the same animals. |
| `PlotStateSweepDaily.jl` / `PlotStateSweepCV.jl` | Plots logL/BIC against K and held-out logL against K. |
| `CrossValidateVTied.jl`, `CompareVTiedVsFull.jl` | Compares tied-drift and full fits: held-out logL, and how the other parameters compensate when `v` is tied. |
| `PlotStateGenerality.jl`, `PlotReviewerStateStructure.jl` | Figures for the response to reviewers (see [below](#response-to-reviewers-analyses)). |

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
| **SF8** — BIC model comparison | `PlotConstrainedBIC.jl`, fed by `FitOneRat.jl` → `MergeBicSummaries.jl` <!-- REVIEW: confirm the constrained refits are what feed SF8. MergeBicSummaries writes results/final_ddmhmms/bic_summary.csv, but PlotConstrainedBIC reads results/bic_summary.csv. --> | `bic_heatmap_from_full.svg`, `bic_change_from_full_bar.svg`, `bic_winners.csv` |
| **SF9** — Drift rate under high contrast | *not in this repository* — the contrast-manipulation DDMs were fit with the [`rddm`](https://github.com/gkane26/rddm) R package (QMPE), as described in the paper's Methods | — |
| **SF10** — Parameter learning in a multi-level DDM | `eDDM_posthoc.jl` §2–§3 | `eddm_elbo_history_by_rat.svg`, `eddm_hyperparams_by_rat.svg` |
| **SF11** — DDM-HMM and GLM-HMM find similar dynamics | `ModelCompFigure.jl` §4–§6 | `model_comparison_lift_heatmap.eps`, `group_null_hist.eps`, `zscores_by_animal.eps` |
| **SF12** — GLM-HMM states align with task structure | from the GLM-HMM fits (`GLM_HMM.jl`) | plotted inline |
| **SF13** — GLM-HMM state structure in an expert rat | `ModelCompFigure.jl` §7 | `null_1065.eps`, `null_Draco.eps`, `logp_by_animal.eps` |

#### Added in revision

<!-- REVIEW: fill in figure numbers (and main vs. supplementary) once the revised manuscript is final. -->

All output paths below are relative to `results/`.

| Figure | Panels | Produced by | Output files |
|---|---|---|---|
| **SF?** — Parameter recovery | A–D true vs. recovered v, B, a₀, τ · E self-transitions · F example session · G state decoding · H within-rat ordering (Kendall τ) · I error vs. dataset size | `ParameterRecovery.jl` → `PlotParameterRecovery.jl` | `parameter_recovery/parameter_recovery.svg`, `recovery_long.csv`, `recovery_summary.csv` |
| **SF?** — States are comparable across animals | A drift vs. accuracy rank · B Kendall τ per parameter · C accuracy vs. v·B | `ExtractStateParameters.jl` → `PlotStateComparability.jl` | `final_ddmhmms/state_comparability.svg` |
| **SF?** — State-conditioned behavior | A psychometric functions · B mean RT by rank · C psychometric slope by rank | `ExtractStateParameters.jl` → `PlotStateBehavior.jl` | `final_ddmhmms/state_behavior.svg` |
| **SF?** — States predict trial initiation | A P(state \| ITI) by decile · B ITI by rank · C trial rate by rank · D P(break > 5 min) by rank · E engagement over a work bout · F specificity control | `ExtractStateParameters.jl` → `StateEngagementITI.jl` → `PlotStateEngagementITI.jl` | `final_ddmhmms/state_engagement_iti.svg`, `state_engagement_*.csv`, `state_bout_profile.csv` |
| **SF?** — Choosing K in session-based animals | logL / ΔlogL / ΔBIC vs. K; held-out logL vs. K; train vs. test gap | `StateSweepDaily.jl` → `MergeStateSweep.jl` → `PlotStateSweepDaily.jl`; `CrossValidateStatesDaily.jl` → `MergeStateSweepCV.jl` → `PlotStateSweepCV.jl` | `ddm_hmm_state_sweep/ll_vs_K_per_rat.svg`, `ll_gain_vs_K.svg`, `bic_vs_K_per_rat.svg`, `state_sweep_winners.csv`; `ddm_hmm_state_sweep/cv/cv_test_ll_vs_K.svg`, `cv_test_ll_gain_vs_K.svg`, `cv_train_vs_test_ll.svg`, `cv_state_sweep_winners.csv` |

Each script also writes a `.png` next to every `.svg`.

### The final K = 4 fits

The revision analyses do **not** read `ddmhmms/`. They read the 18 final
K = 4 tied-full fits in `results/final_ddmhmms/final_ddmhmms/`, one per 24-hour
animal. Those fits come from the constrained Baum–Welch pipeline
(`FitOneRat.jl`) and the direct-gradient pipeline (`FitOneRatGradient.jl`),
which save three different BSON layouts. <!-- REVIEW: say how the fit for each rat was picked (best logL across the two pipelines?). -->
`ExtractStateParameters.jl` and `StateEngagementITI.jl` can load all three
layouts, so the fits don't need to be converted first. To load them from the
older notebooks, run `ConvertConstrainedToDDMHMMFit.jl`, which writes
`results/final_ddmhmms/ddmhmmfit_compat/`.

### Response-to-reviewers analyses

These scripts back the response to reviewers. They are kept here for
transparency, but none of them is a manuscript figure.
<!-- REVIEW: move any of these into the table above if they made it into the paper. -->

| Script | Question it answers | Output |
|---|---|---|
| `PlotStateGenerality.jl` | Do the states recur across animals? States are matched on DDM parameters alone. Includes leave-one-animal-out rank assignment against a shuffle null. | `final_ddmhmms/reviewer_state_generality.svg` |
| `PlotReviewerStateStructure.jl` | Can PCA, k-means or consensus alignment recover a cross-animal state taxonomy without using accuracy? | `final_ddmhmms/reviewer_state_structure.svg` |
| `CrossValidateVTied.jl` | Held-out logL for full vs. tied-drift fits. | `ddm_hmm_constrained/cv_*_fold*.csv` |
| `CompareVTiedVsFull.jl` | When drift is tied across states, how do the other parameters compensate? | `ddm_hmm_constrained/v_tied_compensation/` |
| `DirectGradientDDMHMM.jl` | Single-rat check of the direct-gradient fitter. | `ddm_hmm_constrained/direct_gradient_<rat>/` |

`results/ddm_hmm_constrained/` and `results/ddm_hmm_gradient/` are gitignored,
so the last three only have outputs on a machine where they have been run.

### Cluster scripts

The long fits were run on an SGE cluster. The wrappers live in
[`cluster/`](../cluster/). Each one `cd`s to a hard-coded project path on the
original cluster, so edit that line and the `#$ -P` project before running
`qsub cluster/<wrapper>.sh`.

| Wrapper | Runs | Array over |
|---|---|---|
| `ddmhmms.sh` | `FitDDMHMMs.jl` | — |
| `run_constrainedDDMHMM.sh` / `run_constrainedDDMHMM_array.sh` | `FitConstrainedDDMHMMs.jl` / `FitOneRat.jl` | rats |
| `run_directGradient.sh` / `run_directGradient_array.sh` | `DirectGradientDDMHMM.jl` / `FitOneRatGradient.jl` | rats |
| `submit_directGradient_subset.sh` | `FitOneRatGradient.jl` | a subset of rats |
| `run_parameterRecovery.sh` | `ParameterRecovery.jl` | recovery tasks |
| `run_stateSweepDaily_array.sh` | `StateSweepDaily.jl` | daily rats |
| `run_crossValidateStatesDaily_array.sh` | `CrossValidateStatesDaily.jl` | rat × fold |
| `run_crossValidateVTied_array.sh` | `CrossValidateVTied.jl` | rats |

After an array job finishes, run the matching `Merge*.jl` script to combine the
per-task outputs.

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
FitDDMHMMs.jl            # fits every animal; writes ddmhmms/*.bson   (slow; see cluster/ddmhmms.sh)
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

The scripts added in revision form a second tree, rooted at the final K = 4
fits:

```
FitOneRat.jl / FitOneRatGradient.jl   # final K=4 fits -> results/final_ddmhmms/final_ddmhmms/  (cluster)
   |
   +-- MergeBicSummaries.jl -> PlotConstrainedBIC.jl       -> SF8
   +-- ExtractStateParameters.jl                           # writes state_*_long.csv
   |      +-- PlotStateComparability.jl                    -> SF? (comparability)
   |      +-- PlotStateBehavior.jl                         -> SF? (state behavior)
   |      +-- StateEngagementITI.jl -> PlotStateEngagementITI.jl  -> SF? (engagement / ITI)
   +-- ParameterRecovery.jl -> PlotParameterRecovery.jl    -> SF? (recovery)   (cluster)

StateSweepDaily.jl -> MergeStateSweep.jl -> PlotStateSweepDaily.jl                  -> SF? (daily K)  (cluster)
CrossValidateStatesDaily.jl -> MergeStateSweepCV.jl -> PlotStateSweepCV.jl          -> SF? (daily K)  (cluster)
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
