# Data dictionary

This file documents the behavioral data in this folder, and — for convenience, since
they are derived from it — the fitted models in [`ddmhmms/`](../ddmhmms/) and the
derived summary tables in [`results/`](../results/).

Each row of the behavioral tables is a single trial of a visual
evidence-accumulation task: the animal watches flashes on the left and right and
chooses the side with more flashes. Animals ran in an autonomous facility, some on
standard daytime sessions and some in continuous 24-hour sessions (the `daily`
column distinguishes these). The preprocessed table is produced from the raw one by
[`notebooks/DataPreprocess.jl`](../notebooks/DataPreprocess.jl).

> **Archival copy.** The trial-level behavioral data are also deposited on Zenodo,
> which is the citable version of record:
> **[10.5281/zenodo.22148167](https://doi.org/10.5281/zenodo.22148167)** (CC BY 4.0).
> The deposit carries the same two CSV files together with a standalone codebook.

---

## Files in this folder

| File | Description |
|------|-------------|
| `rat_data.csv.gz` | Raw trial-level export. `flashes_left`/`flashes_right` are per-bin **bit strings** (one character per stimulus bin, `1` = flash). |
| `processed_rat_data.csv.gz` | Cleaned/derived table used by all downstream analyses. `flashes_*` are collapsed to **counts**, plus added derived columns (`delta_flashes`, `choose_right`, `correct`). |
| `rat_data.zip` | Zipped copy of the same raw CSV as `rat_data.csv.gz`. Note that `DataPreprocess.jl` reads an uncompressed `data/rat_data.csv`, which is not committed — decompress either archive before rerunning preprocessing. |

### `rat_data.csv.gz` (raw) columns

| Column | Type | Meaning |
|--------|------|---------|
| *(unnamed index)* | int | Row index. |
| `name` | string | Animal identifier (name or numeric ID). |
| `trial` | int | Trial counter **within a recording session**. Restarts at 1 with each new session — usually once per day, occasionally more than once. Not unique per animal, and not a unique key; combine `name` and `trial_datetime` to identify a trial uniquely. |
| `trial_datetime` | datetime | Timestamp of the trial (`yyyy-mm-dd HH:MM:SS`), local time, not adjusted for daylight saving. |
| `choice` | string | Side chosen: `left`, `right`, or `omission`. |
| `outcome` | string | Trial outcome: `correct`, `error`, or `omission`. |
| `rt` | float | Reaction time in seconds, from center-port exit to side-port entry (range 0–8.1 s). |
| `init_time` | float | Time in seconds from the end of the previous trial to initiation of this one. The task is self-paced, so values range from <0.001 s to ~69,000 s (~19 h). |
| `correct_side` | string | Rewarded side for the trial (`left` / `right`). |
| `flashes_left` | string | Per-bin bit string of left-side flashes (`1` = flash in that bin). |
| `flashes_right` | string | Per-bin bit string of right-side flashes. |
| `sex` | string | Animal sex (`male` / `female`). |
| `daily` | string | Session regime — exactly two values: `daily` (session-based; 285,431 trials, 14 animals) and `24 hr` (live-in; 600,876 trials, 18 animals). |

The raw table has 886,307 rows spanning 2021-05-02 to 2022-04-19, from 32 animals.

**Bit-string structure.** The left and right strings are always equal in length and
exactly complementary — precisely one side flashes in each 100 ms bin. Lengths run
0–80 characters, i.e. 0–8.0 s of evidence (median 10 bins = 1.0 s). 1,253 rows have
empty strings (no stimulus bins); these become `0` counts after preprocessing.

### `processed_rat_data.csv.gz` (preprocessed) columns

All raw columns above, with `flashes_left`/`flashes_right` as **integer counts**
instead of bit strings, plus:

| Column | Type | Meaning |
|--------|------|---------|
| `Column1` | int | The **original raw row index**, carried through unchanged. It is therefore non-contiguous in this file — the gaps are the dropped omission trials — and can be used to join back to the raw table. |
| `delta_flashes` | int | Right minus left flash count; the signed stimulus evidence (Δflashes). |
| `choose_right` | int | `1` if the animal chose right, else `0`. |
| `correct` | int | `1` if the choice was correct, else `0`. |

### How the preprocessed table is derived

`processed_rat_data.csv.gz` is produced from `rat_data.csv.gz` by
[`notebooks/DataPreprocess.jl`](../notebooks/DataPreprocess.jl) via exactly four
operations:

1. `flashes_left` / `flashes_right` are replaced by the count of `1`s in each bit string.
2. The 10,880 omission trials (`choice` and `outcome` both `omission`) are dropped,
   taking the table from 886,307 to 875,427 rows. This is the **only** filtering
   applied — no reaction-time cutoffs, and no exclusion of animals, sessions, or dates.
3. `delta_flashes`, `choose_right`, and `correct` are added.
4. The original row index is carried through as `Column1`.

Every other column is identical between the two files, value for value.

> **Animals:** the preprocessed table contains 32 animals. The DDM-HMM analyses in
> [`ddmhmms/`](../ddmhmms/) are fit on the subset that ran continuous 24-hour sessions
> (18 animals; the `R/` GAMs likewise filter to 24-hour sessions). Those 18 animals
> were run as two successive cohorts of nine — identified by name (cohort 1, recorded
> May and July 2021) and by number (cohort 2, recorded November 2021 to April 2022) —
> under identical housing, training, and testing procedures.

---

## `ddmhmms/`

Fitted DDM-HMM posteriors, one file per animal, serialized with
[`BSON.jl`](https://github.com/JuliaIO/BSON.jl).

**Naming convention:** `{animal}_K{K}_{tying}_compat.bson` — e.g.
`Denna_K4_tied-full_compat.bson` is animal *Denna*, `K = 4` latent states, the
`tied-full` parameterization, saved in a compatibility format.

Each file holds the fitted model for one animal: the latent-state structure, the
drift-diffusion parameters for each state, and the state transition matrix. Load in
Julia with `using BSON; model = BSON.load("ddmhmms/Denna_K4_tied-full_compat.bson")`
and inspect the returned dictionary's keys.

---

## `results/`

Derived summary tables and intermediate outputs.

| File | Description |
|------|-------------|
| `bic_summary.csv` | BIC for every fitted model configuration. Columns: `rat`, `K` (states), `tied` (parameterization), `n_trials`, `n_params`, `logL`, `bic`. |
| `bic_winners.csv` | Best-scoring configuration per animal. Columns: `rat`, `winning_tied`, `winning_bic`, `ΔBIC_to_next`, `n_trials`. |
| `FINAL_fit3.csv` | Final selected fits and cross-validation records. Columns include `K`, `λ`, `test_LLs`, `ntrials_sequence`, `days`, `trials`, paths to per-day posterior state (`gamma_paths`) and time (`time_paths`) files, transition matrix `A`, emission params `B`, initial distribution `πₖ`, and per-trial `outcome`/`choice` and `rat`. |
| `eddm_hyperparams_by_rat.csv` | Extended-DDM (eDDM) fitted hyperparameters per animal. Columns: `rat_name`, per-parameter posterior means `m_u{B,τ,v,a0}`, log posterior SDs `logσ0_u{B,τ,v,a0}`, and group means `{B,τ,v,a0}_group_mean`. |
| `eddm_elbo_history_by_rat.csv` | ELBO optimization trace per animal. Columns: `rat_name`, `iter`, `elbo`. |
| `best_gammas2.jld2` | Best-fit posterior state responsibilities (γ), stored as a [JLD2](https://github.com/JuliaIO/JLD2.jl) file. |

Added in revision. Scripts are listed in [`notebooks/README.md`](../notebooks/README.md#added-in-revision).

| Path | Description |
|------|-------------|
| `final_ddmhmms/final_ddmhmms/` | Final K = 4 tied-full DDM-HMM fits, one BSON per 24-hour animal (18). All the revision analyses read these. |
| `final_ddmhmms/ddmhmmfit_compat/` | The same fits, re-saved in the `ddmhmms/` compatibility format. |
| `final_ddmhmms/state_parameters_long.csv` | One row per rat × state. Columns: `rat`, `state`, DDM parameters `B`, `v`, `a0`, `tau`, `occupancy`, self-transition `p_self`, expected `dwell` (trials), posterior-weighted `acc`, `rt_mean`, `rt_sd`, `p_right`, `abs_df`, `init_time`, and the rat's `n_trials` and `logL`. |
| `final_ddmhmms/state_{psychometric,chronometric,transitions}_long.csv`, `state_psychometric_slopes.csv` | Per-state psychometric and chronometric curves, transition matrices and psychometric slopes, written by `ExtractStateParameters.jl`. |
| `final_ddmhmms/state_engagement_{summary,deciles,stats}.csv`, `state_bout_profile.csv` | Inputs to the trial-initiation (ITI) figure, written by `StateEngagementITI.jl`. |
| `final_ddmhmms/bic_comparison_results/` | Per-rat BIC CSVs from the constrained refits. `MergeBicSummaries.jl` combines them. |
| `parameter_recovery/recovery_summary.csv` | One row per recovery task: `rat`, `rep`, session fraction `frac`, `n_sessions`, `n_trials`, decoding accuracy of the recovered and true models, and logL under the true, recovered, truth-initialised and Baum–Welch-only fits. |
| `parameter_recovery/recovery_scaling.csv` | Per parameter: within-rat power-law exponent `beta` of scaled recovery error vs. trials, rat-bootstrap 95% CI (`lo`, `hi`), intercept `alpha`, `n_rats`. |
| `parameter_recovery/recovery_long.csv` | One row per task × state × parameter, comparing true and recovered values. |
| `ddm_hmm_state_sweep/` | K = 1…5 sweeps on the session-based animals: raw logL/BIC (`state_sweep_summary.csv`) and 5-fold CV (`cv/cv_state_sweep_summary.csv`), with per-fold BSONs in `cv/`. |

Parameter symbols follow the following convention: `v` drift rate, `B` boundary separation, 
`a0` starting-point bias, `τ` non-decision time.
