# Notebooks

Pluto notebooks and headless scripts for the analyses and figures in the paper.
Fits are produced by the scripts; diagnostics, statistics and figure panels run
in Pluto.

## 1. Install Julia

Developed against **Julia 1.11.7**; any 1.11.x should work. Install with
[`juliaup`](https://github.com/JuliaLang/juliaup):

```bash
# macOS / Linux
curl -fsSL https://install.julialang.org | sh

# Windows (PowerShell)
winget install julia -s msstore
```

Then:

```bash
juliaup add 1.11
juliaup default 1.11
```

## 2. Install Pluto

Pluto is a dependency of this folder's `Project.toml`, so instantiating the
environment (next section) installs it. Or add it globally:

```julia
using Pkg
Pkg.add("Pluto")
```

## 3. Set up the environment

From the **repository root**:

```bash
julia --project=notebooks -e 'using Pkg; Pkg.instantiate()'
```

`Manifest.toml` is not committed and `Project.toml` has no `[compat]` bounds,
so you get current releases, possibly newer than the paper's. The first run
takes several minutes.

`Project.toml` has two path dependencies:

- `BehaviorModels`, the package at the root of this repository (resolves
  automatically via the `[sources]` table).
- `DriftDiffusionModels`, expected at `../../DriftDiffusionModels.jl`, i.e.
  next to this repository. Clone
  [DriftDiffusionModels.jl](https://github.com/depasquale-lab/DriftDiffusionModels.jl)
  there, or edit the `[sources]` path.

## 4. Launch a Pluto notebook

From the repo root:

```bash
julia --project=notebooks -e 'using Pluto; Pluto.run()'
```

Open any `.jl` under `notebooks/` from the file picker, or directly:

```bash
julia --project=notebooks -e 'using Pluto; Pluto.run(notebook="notebooks/PopulationDDMHMM.jl")'
```

## 5. Notebook map

| Notebook | Purpose |
|----------|---------|
| `DataPreprocess.jl` | Build the cleaned trial-level dataset used by everything downstream. |
| `FitDDMHMMs.jl` | Headless DDM-HMM fitter (also driven by `cluster/ddmhmms.sh`). |
| `CrossValidationLoop.jl` | Cross-validated state-count selection. |
| `PlotConstrainedBIC.jl` | BIC summaries and state-count selection figure. |
| `GLM_HMM.jl`, `HMM_DDM.jl` | GLM-HMM fits and the DDM-HMM head-to-head setup. |
| `PopulationDDMHMM.jl` | Population-level analyses, parameter contrasts, RM-ANOVA, post-hocs. |
| `PosthocDDMHMM.jl` | Per-rat post-hoc analyses of fitted DDM-HMMs. |
| `ModelCompFigure.jl` | GLM-HMM vs DDM-HMM alignment, NMI permutation tests, BH-FDR. |
| `MiscFigures.jl` | Miscellaneous supplementary panels. |
| `Rat_Daily.jl` | Per-rat and daily exploratory views. |
| `eDDM.jl`, `eDDM_posthoc.jl` | Extended-DDM variants and follow-up. |

### Analysis scripts

Plain Julia scripts: `julia --project=notebooks notebooks/<Script>.jl` from the
repository root. Slow fits run as SGE array jobs via the
[cluster wrappers](#cluster-scripts).

| Script | Purpose |
|--------|---------|
| `FitConstrainedDDMHMMs.jl` | K = 4 DDM-HMM fitter for each tying config (`full`, `τ`, `a₀`, `τ+a₀`, `v`, `B`), Baum–Welch, scored by BIC. Also the shared fitting code other scripts `include`. |
| `FitOneRat.jl` | Runs `FitConstrainedDDMHMMs.jl` for one rat, as one array task. |
| `FitOneRatGradient.jl` | Same configs as above, fit by direct L-BFGS on the marginal likelihood instead of Baum–Welch. |
| `DirectGradientDDMHMM.jl` | Single-rat test of the direct-gradient fitter (30 restarts). |
| `MergeBicSummaries.jl` | Combines the per-rat BIC CSVs into `results/final_ddmhmms/bic_summary.csv`. |
| `ConvertConstrainedToDDMHMMFit.jl` | Re-saves the final constrained fits in the `ddmhmms/*_compat.bson` format that the older notebooks read. |
| `ExtractStateParameters.jl` | Per-rat, per-state DDM parameters, posterior-weighted behavior and psychometric/chronometric curves from the final K = 4 fits. Every state-level figure reads its output. |
| `PlotStateComparability.jl` | Cross-animal comparability of states: drift against accuracy rank, within-rat Kendall τ, and accuracy against v·B. |
| `PlotStateBehavior.jl` | Behavior conditioned on state: psychometric functions, RT by rank, and psychometric slope by rank. |
| `StateEngagementITI.jl` | Whether the states predict trial initiation (ITI, trial rate, breaks), which the model never sees. |
| `PlotStateEngagementITI.jl` | Figure for the analysis above. |
| `StateOccupancyByHour.jl` | Posterior occupancy of each state by hour from lights-on, with the paper-wide accuracy rank. |
| `PlotFigure5.jl` | Figure 5 (composed SVG plus single panels in `figure5_panels/`): occupancy by hour, dark/feeding enrichment, parameters by rank with RM-ANOVA + Holm post-hocs, rat × rank heatmaps. |
| `ParameterRecovery.jl` | Parameter recovery: simulate from each rat's final fit, refit blind, at full size and 10/25/50 % of sessions. |
| `PlotParameterRecovery.jl` | Parameter-recovery figure and summary tables. |
| `StateSweepDaily.jl`, `MergeStateSweep.jl` | Sweeps K = 1…5 on the session-based ("daily") animals and records full-data logL, AIC and BIC. |
| `CrossValidateStatesDaily.jl`, `MergeStateSweepCV.jl` | Session-blocked 5-fold cross-validation over K = 1…5 for the same animals. |
| `PlotStateSweepDaily.jl`, `PlotStateSweepCV.jl` | Plots logL/BIC against K and held-out logL against K. |
| `CrossValidateVTied.jl`, `CompareVTiedVsFull.jl` | Compares tied-drift and full fits: held-out logL, and how the other parameters compensate when `v` is tied. |
| `eDDMExact.jl` | Exact maximum-marginal-likelihood refit of the per-rat eDDM, replacing the variational fit in `eDDM.jl`. Writes `results/eddm_exact/`; `eDDM_posthoc.jl` reads it with `fit_source = :exact`. Merge with `eDDMExact.jl merge`. |
| `SessionSplitDDM.jl` | Within- vs. between-session control: rat-level DDM, per-session DDMs and K = 4 DDM-HMM fit on the first 80% of each session, scored on the last 20%. Writes `results/session_cohort/split_<group>/`. |
| `SessionCohortSummary.jl` | Joins held-out results for both cohorts (DDM-HMM K = 1…5, shuffled mixture, multilevel DDM, within-session split), paired Wilcoxon tests, within-session switching from held-out Viterbi paths. Writes `results/session_cohort/`. |
| `SessionCohortACF.jl` | Within-session RT autocorrelation, real vs. simulated from each rat's full-data DDM-HMM and multilevel DDM, both cohorts. Writes `results/session_cohort/rt_acf_by_rat.csv`, `rt_acf_error.csv`. |
| `PlotSessionCohortFigure.jl` | Session-vs-24 h figure: held-out gain vs. each competitor, and RT autocorrelation model vs. data. Writes the figure and per-panel SVGs to `results/session_cohort/panels/`. |
| `PlotStateGenerality.jl`, `PlotStateStructure.jl` | Cross-animal state matching figures (see [below](#supplementary-analyses)). |

## 6. Reproducing the published figures

Schematic panels were drawn in BioRender or Illustrator and have no code.

### Main figures

| Figure | Panels | Produced by | Output files |
|---|---|---|---|
| **Fig. 1**: Live-in operant facility and task | A–C | *schematics, no code* | none |
| | D | `MiscFigures.jl` §2 | `rat_trial_time_histogram.svg` |
| | E | `MiscFigures.jl` §3 | `rat_daily_acc.svg` |
| **Fig. 2**: Individual variability and diurnal modulation | A–C | `R/GAMS_Flashes_24Hour.R` | `ranked_acc.svg`, `ranked_rt.svg`, `ranked_trials.svg` |
| | D–F | `R/GAMS_Flashes_24Hour.R` | `flashes_acc_vs_tod_marginal.pdf`, `flashes_rt_vs_tod_marginal.pdf`, `flashes_trials_rate_marginal.pdf` |
| **Fig. 3**: Accurate account of behavior | A (cross-validation) | `CrossValidationLoop.jl` | writes `results/`; plotted inline |
| | B–D (RT fit, Q–Q, accuracy) | `PopulationDDMHMM.jl` §12 | `rt_distributions_by_rat.svg`, `qq_population_rts.svg`, `acc_pred_population_rankcolored.svg` |
| **Fig. 4**: Fit for an individual rat | A–G | `PosthocDDMHMM.jl`, with `rat = "Remy"` | `posthoc_remy_*.svg` |
| **Fig. 5**: States across the 24-hour cycle | all | `PopulationDDMHMM.jl` §8–§10 | `population_state_occupancy.svg`, `population_enrichment.svg`, `parameter_boxswarm_by_accRank_population.svg` |
| **Fig. 6**: Trial-to-trial RT correlations | A (parameter ACFs) | `eDDM_posthoc.jl` §4 (fits from `eDDMExact.jl`) | `eddm_exact/eddm_trial_param_acf.svg` |
| | B (RT ACF vs. both models) | `eDDM_posthoc.jl` §5–§6 | `eddm_exact/eddm_rt_acf.svg`, `eddm_exact/eddm_rt_acf_ppc.svg` |
| **Fig. 7**: Short-timescale fluctuations | A, B (RT distributions) | `Rat_Daily.jl` §5, §4 | plotted inline |
| | C (posterior decoding) | `Rat_Daily.jl` §3 | plotted inline |
| | D (population ACF) | `Rat_Daily.jl` §2 | `rat_daily_acf_ppc.svg` |

### Supplementary figures

| Figure | Produced by | Output files |
|---|---|---|
| **SF1**: Training procedure and dataset summary | *schematic + summary table, no code* | none |
| **SF2**: Individual GAM fits, reaction time | `R/GAMS_Flashes_24Hour.R` | `flashes_rt_facet.svg` |
| **SF3**: Individual GAM fits, accuracy | `R/GAMS_Flashes_24Hour.R` | `flashes_acc_facet.svg` |
| **SF4**: Individual GAM fits, trial production | `R/GAMS_Flashes_24Hour.R` | `flashes_trials_facet.svg` |
| **SF5**: RT distributions | `PopulationDDMHMM.jl` §12 | `rt_distributions_by_rat.svg`, `accuracy_fit_by_rat.svg` |
| **SF6**: Example fit for an expert rat | `PosthocDDMHMM.jl`, with `rat = "1062"` | `posthoc_1062_*.svg` |
| **SF7**: DDM parameter regressions | `PopulationDDMHMM.jl` §6–§7 | `beta_forest_population.svg`, `beta_forest_population_rt.svg` |
| **SF8**: BIC model comparison | `PlotConstrainedBIC.jl`, after `FitOneRat.jl` and `MergeBicSummaries.jl` <!-- REVIEW: confirm the constrained refits are what feed SF8. MergeBicSummaries writes results/final_ddmhmms/bic_summary.csv, but PlotConstrainedBIC reads results/bic_summary.csv. --> | `bic_heatmap_from_full.svg`, `bic_change_from_full_bar.svg`, `bic_winners.csv` |
| **SF9**: Drift rate under high contrast | *not in this repository*; the contrast-manipulation DDMs were fit with the [`rddm`](https://github.com/gkane26/rddm) R package (QMPE), as described in the paper's Methods | none |
| **SF10**: Parameter learning in a multi-level DDM | `eDDM_posthoc.jl` §2–§3 | `eddm_exact/eddm_elbo_history_by_rat.svg` (fit summary for exact fits), `eddm_exact/eddm_hyperparams_by_rat.svg` |
| **SF11**: DDM-HMM and GLM-HMM find similar dynamics | `ModelCompFigure.jl` §4–§6 | `model_comparison_lift_heatmap.eps`, `group_null_hist.eps`, `zscores_by_animal.eps` |
| **SF12**: GLM-HMM states align with task structure | from the GLM-HMM fits (`GLM_HMM.jl`) | plotted inline |
| **SF13**: GLM-HMM state structure in an expert rat | `ModelCompFigure.jl` §7 | `null_1065.eps`, `null_Draco.eps`, `logp_by_animal.eps` |

#### Script-generated figures

<!-- REVIEW: fill in figure numbers (and main vs. supplementary) once the manuscript is final. -->

Scripts are listed in run order. Output paths are relative to `results/`.

| Figure | Produced by | Output files |
|---|---|---|
| Parameter recovery | `ParameterRecovery.jl`, `PlotParameterRecovery.jl` | `parameter_recovery/parameter_recovery.svg`, `recovery_long.csv`, `recovery_summary.csv`, `recovery_scaling.csv` |
| States are comparable across animals | `ExtractStateParameters.jl`, `PlotStateComparability.jl` | `final_ddmhmms/state_comparability.svg` |
| State-conditioned behavior | `ExtractStateParameters.jl`, `PlotStateBehavior.jl` | `final_ddmhmms/state_behavior.svg` |
| States predict trial initiation | `ExtractStateParameters.jl`, `StateEngagementITI.jl`, `PlotStateEngagementITI.jl` | `final_ddmhmms/state_engagement_iti.svg`, `state_engagement_*.csv`, `state_bout_profile.csv` |
| Choosing K in session-based animals | `StateSweepDaily.jl`, `MergeStateSweep.jl`, `PlotStateSweepDaily.jl`; `CrossValidateStatesDaily.jl`, `MergeStateSweepCV.jl`, `PlotStateSweepCV.jl` | `ddm_hmm_state_sweep/ll_vs_K_per_rat.svg`, `ll_gain_vs_K.svg`, `bic_vs_K_per_rat.svg`, `state_sweep_winners.csv`; `ddm_hmm_state_sweep/cv/cv_test_ll_vs_K.svg`, `cv_test_ll_gain_vs_K.svg`, `cv_train_vs_test_ll.svg`, `cv_state_sweep_winners.csv` |
| DDM-HMM vs. multilevel DDM in session and 24 h data | `SessionCohortSummary.jl`, `SessionCohortACF.jl`, `PlotSessionCohortFigure.jl` | `session_cohort/session_cohort_figure.svg`, `session_cohort/panels/*.svg`, `heldout_stats.csv`, `rt_acf_r2.csv` |

Each script also writes a `.png` next to every `.svg`.

### The final K = 4 fits

The analysis scripts read the 18 final K = 4 tied-full fits in
`results/final_ddmhmms/final_ddmhmms/`, not `ddmhmms/`. They come from
`FitOneRat.jl` (Baum–Welch) and `FitOneRatGradient.jl` (direct gradient), in
three BSON layouts, all of which the scripts load directly.
<!-- REVIEW: say how the fit for each rat was picked (best logL across the two pipelines?). -->
`ConvertConstrainedToDDMHMMFit.jl` rewrites them for the older notebooks
(`results/final_ddmhmms/ddmhmmfit_compat/`).

### Supplementary analyses

None of these is a manuscript figure.
<!-- REVIEW: move any of these into the table above if they made it into the paper. -->

| Script | Purpose | Output |
|---|---|---|
| `PlotStateGenerality.jl` | Matching states across animals: every label-free route agrees with accuracy rank in 16-17/18 animals (absolute-parameter control 7/18); PCA on absolute vs within-animal parameters (accuracy R² 0.37 vs 0.70); within-animal PC1 vs accuracy rank. | `final_ddmhmms/state_generality.svg` |
| `PlotStateStructure.jl` | PCA, k-means and consensus alignment of states without using accuracy. Rank structure appears only with within-animal z-scoring. | `final_ddmhmms/state_structure.svg` |
| `CrossValidateVTied.jl` | Held-out logL for full vs. tied-drift fits. | `ddm_hmm_constrained/cv_*_fold*.csv` |
| `CompareVTiedVsFull.jl` | How the other parameters compensate when drift is tied across states. | `ddm_hmm_constrained/v_tied_compensation/` |
| `DirectGradientDDMHMM.jl` | Single-rat check of the direct-gradient fitter. | `ddm_hmm_constrained/direct_gradient_<rat>/` |
| `FitTimeOfDayDDM.jl` | Time-of-day baseline: single-state DDM with Fourier-in-clock-time parameters (H = 0…3; `VARY_TAU=1` also varies τ), same folds as the 24 h CV, plus an all-data fit for BIC. Merge with `FitTimeOfDayDDM.jl merge`. | `tod_ddm/` (τ fixed), `tod_ddm_vary_tau/` (τ varying): `tod_ddm_summary.csv` |
| `CompareTimeOfDayDDM.jl` | Held-out logL and BIC of the best time-of-day DDM vs. the K = 4 DDM-HMM, per rat. Set `VARY_TAU=1` for the τ-varying fits. | `tod_ddm{,_vary_tau}/tod_vs_hmm_{cv,bic}.csv` |
| `TimeOfDayFloorCheck.jl` | Repeats the held-out comparison without trials that hit the WFPT density floor (RT ≤ τ). | `tod_ddm_vary_tau/floor_check{,_by_fold}.csv` |
| `PlotTimeOfDayComparison.jl` | Held-out gain over the DDM for the time-of-day DDM and the DDM-HMM, with and without floored trials, and the all-data BIC difference. | `tod_ddm_vary_tau/tod_vs_hmm.svg` |
| `TimeOfDayParameterCurves.jl` | DDM parameters by hour from lights-on: the DDM-HMM's posterior-weighted trial parameters vs. the time-of-day DDM's curves, and the fraction of the HMM's trial-level parameter variance explained by hour. | `tod_ddm_vary_tau/param_curves_long.csv`, `param_variance_by_hour.csv` |
| `PlotTimeOfDayParameterCurves.jl` | Figure for the analysis above: one example rat, then all rats with each curve centred on its own daily mean. | `tod_ddm_vary_tau/param_curves.svg` |
| `SessionCohortPredictRT.jl` | One-step-ahead E[RT] on held-out sessions: DDM-HMM, multilevel DDM, running mean. Not in the figure: the running mean matches the DDM-HMM on raw R²; the DDM-HMM wins within session. | `session_cohort/predict_rt_by_rat.csv` |

`results/ddm_hmm_constrained/` and `results/ddm_hmm_gradient/` are gitignored,
so `CrossValidateVTied.jl`, `CompareVTiedVsFull.jl` and `DirectGradientDDMHMM.jl` only have
outputs on a machine where they have been run. The time-of-day per-rat fits
(`tod_ddm*/per_rat/`) are gitignored too; the merged summaries are tracked.

### Cluster scripts

SGE wrappers in [`cluster/`](../cluster/). Each `cd`s to a hard-coded project
path; edit that and the `#$ -P` project before `qsub cluster/<wrapper>.sh`.

| Wrapper | Runs | Array over |
|---|---|---|
| `ddmhmms.sh` | `FitDDMHMMs.jl` | none |
| `run_constrainedDDMHMM.sh`, `run_constrainedDDMHMM_array.sh` | `FitConstrainedDDMHMMs.jl`, `FitOneRat.jl` | rats |
| `run_directGradient.sh`, `run_directGradient_array.sh` | `DirectGradientDDMHMM.jl`, `FitOneRatGradient.jl` | rats |
| `submit_directGradient_subset.sh` | `FitOneRatGradient.jl` | a subset of rats |
| `run_parameterRecovery.sh` | `ParameterRecovery.jl` | recovery tasks |
| `run_stateSweepDaily_array.sh` | `StateSweepDaily.jl` | daily rats |
| `run_crossValidateStatesDaily_array.sh` | `CrossValidateStatesDaily.jl` | rat × fold |
| `run_crossValidateStates24hr_array.sh` | `CrossValidateStatesDaily.jl` (`GROUP=24hr`) | rat × fold |
| `run_crossValidateVTied_array.sh` | `CrossValidateVTied.jl` | rats |
| `run_eDDMExact_array.sh` | `eDDMExact.jl` | rats |
| `run_eDDMExactDaily_array.sh` | `eDDMExact.jl` (`GROUP=daily`) | session rats |
| `run_eDDMExactCV_{24hr,daily}_array.sh` | `eDDMExact.jl cv` | rat × fold |
| `run_cvShuffled_{24hr,daily}_array.sh` | `CrossValidateStatesDaily.jl` (`SHUFFLE=1`, K = 4) | rat × fold |
| `run_sessionSplit_{24hr,daily}_array.sh` | `SessionSplitDDM.jl` | rats |
| `run_timeOfDayDDM_array.sh` | `FitTimeOfDayDDM.jl` (set `VARY_TAU=1` for the τ-varying fits) | rats |
| `run_sessionCohortFinalize.sh` | all session-cohort merges, then `SessionCohortSummary.jl`, `SessionCohortACF.jl` and `PlotSessionCohortFigure.jl` | none (submit with `-hold_jid` on the jobs above) |

After an array job finishes, run the matching `Merge*.jl` script to combine the
per-task outputs.

### Two figures, one notebook

`PosthocDDMHMM.jl` makes both Figure 4 and SF6, set by one variable near the top:

```julia
rat = "1062"     # Supplementary Figure 6 (expert rat)
# rat = "Remy"   # Figure 4 (representative rat)
```

Outputs go to `results/posthoc_<rat>_<panel>.svg`.

### Order of operations

Fitting must precede the figure notebooks. From a clean checkout:

1. `DataPreprocess.jl` builds the processed table (already committed; rerun only if needed).
2. `FitDDMHMMs.jl` fits every animal and writes `ddmhmms/*.bson` (slow; see `cluster/ddmhmms.sh`).
3. `CrossValidationLoop.jl` does state-count selection and writes `results/`.
4. Then, in any order: `PosthocDDMHMM.jl` (Fig. 4, SF6), `PopulationDDMHMM.jl`
   (Fig. 3B–D, Fig. 5, SF5, SF7), `ModelCompFigure.jl` (SF11, SF13),
   `eDDM_posthoc.jl` (Fig. 6, SF10; needs `eDDMExact.jl` first), `Rat_Daily.jl`
   (Fig. 7), `PlotConstrainedBIC.jl` (SF8), `MiscFigures.jl` (Fig. 1D, 1E).

The analysis scripts form a second pipeline, rooted at the final K = 4 fits:

1. `FitOneRat.jl` or `FitOneRatGradient.jl` (cluster) write the final K = 4 fits
   to `results/final_ddmhmms/final_ddmhmms/`.
2. `MergeBicSummaries.jl`, then `PlotConstrainedBIC.jl` (SF8).
3. `ExtractStateParameters.jl` writes `state_*_long.csv`. Then, in any order:
   `PlotStateComparability.jl`; `PlotStateBehavior.jl`; `StateEngagementITI.jl`
   then `PlotStateEngagementITI.jl`; `StateOccupancyByHour.jl` then
   `PlotFigure5.jl` (Fig. 5).
4. `ParameterRecovery.jl` (cluster), then `PlotParameterRecovery.jl`.
5. `StateSweepDaily.jl` (cluster), `MergeStateSweep.jl`, `PlotStateSweepDaily.jl`.
6. `CrossValidateStatesDaily.jl` (cluster), `MergeStateSweepCV.jl`, `PlotStateSweepCV.jl`.

`PlotConstrainedBIC.jl`, `GLM_HMM.jl`, `HMM_DDM.jl` and parts of `Rat_Daily.jl`
render panels inline in Pluto rather than writing files.

`R/GAMS_Flashes_24Hour.R` is independent of the Julia pipeline and reads
`data/processed_rat_data.csv.gz` directly. It produces Figure 2 and SF2–SF4.

## 7. A note on figures

Notebook output will not match the published panels pixel-for-pixel: those
were exported as SVG/EPS and finished in Illustrator. Numbers, statistics and
geometry are the same. Questions: rsenne at bu dot edu.
