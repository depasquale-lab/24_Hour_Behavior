# Stitch the per-task CV state-sweep CSVs (one per rat x fold) into one summary.
using Pkg
Pkg.activate("notebooks")

using CSV, DataFrames

out_dir = joinpath("results", "ddm_hmm_state_sweep", "cv")
per_task_dir = joinpath(out_dir, "per_task_summaries")

files = filter(f -> endswith(f, ".csv"), readdir(per_task_dir; join=true))
isempty(files) && error("No per-task summaries found in $per_task_dir")

summary = reduce(vcat, [CSV.read(f, DataFrame; types=Dict(:rat => String)) for f in files])
sort!(summary, [:rat, :K, :fold])

out = joinpath(out_dir, "cv_state_sweep_summary.csv")
CSV.write(out, summary)
@info "Merged $(length(files)) per-task files -> $out ($(nrow(summary)) rows)"

# Warn about any (rat, K) cell that is missing folds.
expected = maximum(summary.fold)
for sub in groupby(summary, [:rat, :K])
    nrow(sub) == expected || @warn "incomplete cell" rat = sub.rat[1] K = sub.K[1] n_folds = nrow(
        sub
    )
end

show(summary; allrows=true, allcols=true)
println()
