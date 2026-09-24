using Pkg
Pkg.activate("notebooks")

using CSV
using DataFrames

results_dir = joinpath("results", "final_ddmhmms")
per_task_dir = joinpath(results_dir, "bic_comparison_results")

@assert isdir(per_task_dir) "missing $(per_task_dir) — did the array job run?"

files = sort(filter(f -> endswith(f, ".csv"), readdir(per_task_dir; join=true)))
@assert !isempty(files) "no per-task CSVs in $per_task_dir"

@info "Merging $(length(files)) per-task summaries..."

summary = reduce(vcat, [CSV.read(f, DataFrame) for f in files])
summary.rat = string.(summary.rat)
# summary.K = [x isa AbstractString ? parse(Int, x) : Int(x) for x in summary.K]
# summary.tied = [x isa AbstractString ? parse(Bool, x) : Bool(x) for x in summary.tied]
sort!(summary, [:rat, :K, :tied])

out = joinpath(results_dir, "bic_summary.csv")
CSV.write(out, summary)
@info "Wrote merged summary to $out ($(nrow(summary)) rows)"
