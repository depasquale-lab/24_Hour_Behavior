# Data dictionary

This file documents the behavioral data in [`data/`](data/), the fitted models in
[`ddmhmms/`](ddmhmms/), and the derived summary tables in [`results/`](results/).

Each row of the behavioral tables is a single trial of a visual
evidence-accumulation task: the animal watches flashes on the left and right and
chooses the side with more flashes. Animals ran in an autonomous facility, some on
standard daytime sessions and some in continuous 24-hour sessions (the `daily`
column distinguishes these). The preprocessed table is produced from the raw one by
[`notebooks/DataPreprocess.jl`](notebooks/DataPreprocess.jl).

---

## `data/`

| File | Description |
|------|-------------|
| `rat_data.csv.gz` | Raw trial-level export. `flashes_left`/`flashes_right` are per-bin **bit strings** (one character per stimulus bin, `1` = flash). |
| `processed_rat_data.csv.gz` | Cleaned/derived table used by all downstream analyses. `flashes_*` are collapsed to **counts**, plus added derived columns (`delta_flashes`, `choose_right`, `correct`). |
| `rat_data.zip` | Zipped copy of the raw data. |

### `rat_data.csv.gz` (raw) columns

| Column | Type | Meaning |
|--------|------|---------|
| *(unnamed index)* | int | Row index. |
| `name` | string | Animal identifier (name or numeric ID). |
| `trial` | int | Trial number for that animal. |
| `trial_datetime` | datetime | Timestamp of the trial (`yyyy-mm-dd HH:MM:SS`). |
| `choice` | string | Side chosen (`left` / `right`). |
| `outcome` | string | Trial outcome (`correct` / `error`). |
| `rt` | float | Reaction time (seconds). |
| `init_time` | float | Time to initiate the trial (seconds). |
| `correct_side` | string | Rewarded side for the trial (`left` / `right`). |
| `flashes_left` | string | Per-bin bit string of left-side flashes (`1` = flash in that bin). |
| `flashes_right` | string | Per-bin bit string of right-side flashes. |
| `sex` | string | Animal sex (`male` / `female`). |
| `daily` | string | Session regime — `daily` for standard sessions, `24`/`24h`/etc. for 24-hour sessions. |

### `processed_rat_data.csv.gz` (preprocessed) columns

All raw columns above, with `flashes_left`/`flashes_right` as **integer counts**
instead of bit strings, plus:

| Column | Type | Meaning |
|--------|------|---------|
| `Column1` | int | Row index. |
| `delta_flashes` | int | Right minus left flash count; the signed stimulus evidence (Δflashes). |
| `choose_right` | int | `1` if the animal chose right, else `0`. |
| `correct` | int | `1` if the choice was correct, else `0`. |

> **Animals:** the preprocessed table contains ~32 animals. The DDM-HMM analyses in
> [`ddmhmms/`](ddmhmms/) are fit on the subset that ran continuous 24-hour sessions
> (18 animals; the `R/` GAMs likewise filter to 24-hour sessions).

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

Parameter symbols follow the following convention: `v` drift rate, `B` boundary separation, 
`a0` starting-point bias, `τ` non-decision time.
