# BehaviorModels — Diurnal rhythms of choice

Code, data, and fitted models for:

> **Diurnal rhythms of choice: a novel state-dependent drift diffusion model
> uncovers time-dependent changes in rat decision making.**
> Ryan A. Senne, Hongjie Xia, Helene F. Duebel, Quan Do, Gary A. Kane, James Fourie, Steve Ramirez, Brian DePasquale, Benjamin B. Scott *bioRxiv* (2026).
> https://www.biorxiv.org/content/10.64898/2026.05.25.727672v1


## Overview

Rats working around the clock in an autonomous facility shift in accuracy,
reaction time and willingness to work with time of day. This repository holds
the analysis code behind the paper and its model, a **Drift Diffusion
Model–Hidden Markov Model (DDM-HMM)**: the animal moves between a few latent
states, each with its own drift-diffusion parameters, fit to choices and
reaction times jointly through the Wiener first-passage-time likelihood.

## Repository layout

| Path | What's in it |
|------|--------------|
| [`src/`](src/) | The `BehaviorModels.jl` package: model definitions and utilities. |
| [`notebooks/`](notebooks/) | Pluto notebooks (analyses + figures) and headless fitting scripts. **See [`notebooks/README.md`](notebooks/README.md) for full setup and a per-notebook map.** |
| [`ddmhmms/`](ddmhmms/) | Fitted DDM-HMM posteriors, one `.bson` per animal (`K=4`, tied-full parameterization). |
| [`data/`](data/) | Trial-level behavioral data (raw and preprocessed). Column-by-column codebook in [`data/README.md`](data/README.md). |
| [`results/`](results/) | Derived outputs: BIC/model-comparison summaries, ELBO histories, fitted transition matrices. Documented in [`data/README.md`](data/README.md). |
| [`cluster/`](cluster/) | SGE job wrappers for the long fits (see [`notebooks/README.md`](notebooks/README.md#cluster-scripts)). |
| [`R/`](R/) | `GAMS_Flashes_24Hour.R` — GAMs of accuracy, RT, and trial rate vs. time of day. |

### What's in `src/`

| File | Contents |
|------|----------|
| [`BehaviorModels.jl`](src/BehaviorModels.jl) | Module entry point; includes the files below. |
| [`DataStructs.jl`](src/DataStructs.jl) | `BehaviorTrial` — the per-trial record (Δflashes, choice, correctness, RT). |
| [`TuringModels.jl`](src/TuringModels.jl) | `glmhmm` (Turing model), `BernoulliGLM`/`GLMObs` emissions, AR baseline. |
| [`PreprocessingUtilities.jl`](src/PreprocessingUtilities.jl) | Session tagging, session summaries, sequence-building for the HMMs. |
| [`Utilities.jl`](src/Utilities.jl) | AIC/BIC, cross-validation splits, optimal-accuracy helpers. |

> The DDM-HMM emission/likelihood machinery lives in the companion package
> [`DriftDiffusionModels.jl`](https://github.com/depasquale-lab/DriftDiffusionModels.jl),
> which this package depends on.

## Quickstart

Everything runs in Julia (**1.11.7**). **[`notebooks/README.md`](notebooks/README.md)**
covers installation, the `DriftDiffusionModels.jl` path dependency and Pluto.
From the repository root:

```bash
# 1. Instantiate the analysis environment (downloads pinned package versions)
julia --project=notebooks -e 'using Pkg; Pkg.instantiate()'

# 2. Launch Pluto and open any notebook from notebooks/
julia --project=notebooks -e 'using Pluto; Pluto.run()'
```

A natural reading order: `DataPreprocess.jl` -> `FitDDMHMMs.jl` /
`CrossValidationLoop.jl` -> `PopulationDDMHMM.jl` / `PosthocDDMHMM.jl` ->
`ModelCompFigure.jl`. See the notebook map in `notebooks/README.md`.

## Reproducing figures

**[`notebooks/README.md` §6](notebooks/README.md#6-reproducing-the-published-figures)
maps every panel to the script that produces it** and gives the run order.
Published panels were finished in Illustrator, so notebook output matches them
in numbers and geometry but not pixel-for-pixel.

## Data

Behavioral data and derived model outputs are documented column-by-column in
[`data/README.md`](data/README.md). Raw and preprocessed trial tables are in
[`data/`](data/); fitted models are in [`ddmhmms/`](ddmhmms/); summary tables are in
[`results/`](results/).

The dataset is also archived on Zenodo, which is the citable version of record:
[10.5281/zenodo.22148167](https://doi.org/10.5281/zenodo.22148167) (CC BY 4.0).

## Citation

Please cite the paper (reference at the top). BibTeX to follow once the DOI is final.
<!-- REVIEW: add final BibTeX once confirmed. -->

## License

Released under the MIT License: see [`LICENSE`](LICENSE).

## Contact

Questions about the code or data: **rsenne at bu dot edu**.
