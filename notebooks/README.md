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

## 6. A note on figures

Figures rendered by these notebooks **will not match the published versions
pixel-for-pixel**. The notebook outputs are the analytical source figures
the published panels were exported as vector graphics (SVG / EPS) and then
post-processed in Adobe Illustrator for typography, legends, panel
composition, color harmonization, and layout. The numbers, statistics, and
geometry are the same; the polish is not. If you have any questions, please email
rsenne at bu dot edu.
