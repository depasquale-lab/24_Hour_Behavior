# Stitch the per-task state-sweep CSVs (one per rat) into one summary.
using Pkg
Pkg.activate("notebooks")

using CSV, DataFrames

results_dir = joinpath("results", "ddm_hmm_state_sweep")
per_task_dir = joinpath(results_dir, "per_task_summaries")

files = filter(f -> endswith(f, ".csv"), readdir(per_task_dir; join=true))
isempty(files) && error("No per-task summaries found in $per_task_dir")

summary = reduce(vcat, [CSV.read(f, DataFrame) for f in files])
sort!(summary, [:rat, :K])

out = joinpath(results_dir, "state_sweep_summary.csv")
CSV.write(out, summary)
@info "Merged $(length(files)) per-task files -> $out ($(nrow(summary)) rows)"

show(summary; allrows=true, allcols=true)
println()
