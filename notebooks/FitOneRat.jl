using Pkg
Pkg.activate("notebooks")

include(joinpath(@__DIR__, "FitConstrainedDDMHMMs.jl"))

using BSON: @save

# rat_idx: $SGE_TASK_ID, then $RAT_IDX, then ARGS[1].

function _resolve_rat_idx()
    for k in ("SGE_TASK_ID", "RAT_IDX")
        v = get(ENV, k, "")
        (isempty(v) || v == "undefined") && continue
        return parse(Int, v)
    end
    if !isempty(ARGS)
        return parse(Int, ARGS[1])
    end
    error(
        "FitOneRat.jl: no rat index provided. Set SGE_TASK_ID or RAT_IDX, or pass an integer CLI arg.",
    )
end

const RAT_IDX = _resolve_rat_idx()
@assert 1 <= RAT_IDX <= length(rat_names) "RAT_IDX=$RAT_IDX is out of range 1:$(length(rat_names))"
const THIS_RAT = rat_names[RAT_IDX]

@info "== Task $(RAT_IDX): rat=$THIS_RAT (Julia threads = $(Threads.nthreads())) =="

results_dir = joinpath("results", "ddm_hmm_constrained")
isdir(results_dir) || mkpath(results_dir)

# Per-rat summary (gets stitched together after all tasks finish).
summary_rows = DataFrame(;
    rat=String[],
    K=Int[],
    tied=String[],
    n_trials=Int[],
    n_params=Int[],
    logL=Float64[],
    bic=Float64[],
)

for tied in TIED_CONFIGS
    tied_tag = isempty(tied) ? "full" : join(string.(tied), "_")
    @info "Fitting rat=$THIS_RAT K=$K_STATES tied=$tied_tag with $N_ITERS inits..."

    fit = fit_constrained_ddmhmm_for_rat(RAT_IDX, tied)

    filename = joinpath(results_dir, "$(THIS_RAT)_K$(K_STATES)_tied-$(tied_tag).bson")
    @save filename fit rat = THIS_RAT K_STATES tied

    push!(
        summary_rows,
        (
            rat=THIS_RAT,
            K=K_STATES,
            tied=tied_tag,
            n_trials=fit.n_trials,
            n_params=fit.n_free_params,
            logL=fit.logL,
            bic=fit.bic,
        ),
    )

    @info "  logL=$(round(fit.logL; digits=2))  k=$(fit.n_free_params)  BIC=$(round(fit.bic; digits=2))  →  $filename"
end

# Per-task summary; MergeBicSummaries.jl combines them into bic_summary.csv.
per_task_dir = joinpath(results_dir, "per_task_summaries")
isdir(per_task_dir) || mkpath(per_task_dir)
CSV.write(joinpath(per_task_dir, "$(THIS_RAT)_K$(K_STATES).csv"), summary_rows)
@info "Wrote per-task summary for $THIS_RAT"
