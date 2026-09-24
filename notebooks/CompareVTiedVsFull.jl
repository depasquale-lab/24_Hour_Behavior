using Pkg
Pkg.activate("notebooks")

using Random
using Dates
using Statistics
using Printf
using CSV
using DataFrames
using BSON: @load
using DriftDiffusionModels
using HiddenMarkovModels
using Plots
using StatsPlots

gr()

const FONT_FAMILY = "Helvetica"
default(;
    fontfamily=FONT_FAMILY,
    titlefontfamily=FONT_FAMILY,
    guidefontfamily=FONT_FAMILY,
    tickfontfamily=FONT_FAMILY,
    legendfontfamily=FONT_FAMILY,
)

function savefig_both(p, stem::AbstractString)
    savefig(p, stem * ".png")
    savefig(p, stem * ".svg")
end

#=
Types that must exist in scope so BSON can reconstruct the saved fits.
Mirrors the definitions in FitConstrainedDDMHMMs.jl.
=#

struct ConstrainedDDMHMMFit
    hmm::Any
    tied::Vector{Symbol}
    logL::Float64
    logL_evolution::Vector{Float64}
    n_trials::Int
    n_free_params::Int
    bic::Float64
end

mutable struct TiedPriorHMM{T<:Real} <: HiddenMarkovModels.AbstractHMM
    init::Vector{T}
    trans::Matrix{T}
    dists::Vector{DriftDiffusionModel}
    α_trans::Matrix{T}
    α_init::Vector{T}
    tied::Vector{Symbol}
end

Base.length(hmm::TiedPriorHMM) = length(hmm.init)
HiddenMarkovModels.initialization(hmm::TiedPriorHMM) = hmm.init
HiddenMarkovModels.transition_matrix(hmm::TiedPriorHMM) = hmm.trans
HiddenMarkovModels.obs_distributions(hmm::TiedPriorHMM) = hmm.dists

const DATA_FILE = joinpath("data", "processed_rat_data.csv.gz")

rat_df = CSV.read(DATA_FILE, DataFrame)
rat_df = rat_df[rat_df.daily .== "24 hr", :]
replace!(rat_df[!, :choose_right], 0 => -1)
side_mapping = Dict("right" => 1, "left" => -1)
DataFrames.transform!(
    rat_df,
    :correct_side => ByRow(cs -> get(side_mapping, cs, missing)) => :correct_side_numeric,
)
rat_names_all = String.(unique(rat_df[!, "name"]))

function data_for_rat(rat::AbstractString)
    sub = rat_df[rat_df.name .== rat, :]
    dates = [Date(split(dt)[1]) for dt in sub.trial_datetime]
    unique_dates = sort(unique(dates))

    results_by_date = Vector{Vector{DDMResult}}()
    for date in unique_dates
        idx = findall(dates .== date)
        isempty(idx) && continue
        push!(
            results_by_date,
            [
                DDMResult(rt, ch, st) for (rt, ch, st) in
                zip(sub.rt[idx], sub.choose_right[idx], sub.correct_side_numeric[idx])
            ],
        )
    end
    seq_ends = cumsum([length(seq) for seq in results_by_date])
    all_results = reduce(vcat, results_by_date)
    return all_results, seq_ends
end

"All permutations of 1:n (for n ≤ 5 this is fine; K is typically 4)."
function all_permutations(n::Int)
    n == 1 && return [[1]]
    out = Vector{Int}[]
    for p in all_permutations(n - 1)
        for i in 1:n
            push!(out, vcat(p[1:(i - 1)], n, p[i:end]))
        end
    end
    return out
end

function _fb_gamma(hmm, obs_seq, seq_ends)
    ret = HiddenMarkovModels.forward_backward(hmm, obs_seq; seq_ends=seq_ends)
    return _extract_gamma(ret)
end

function _extract_gamma(ret)
    if ret isa AbstractMatrix
        return ret
    elseif ret isa Tuple
        return _extract_gamma(ret[1])
    elseif hasproperty(ret, :γ)
        return getfield(ret, :γ)
    else
        error("don't know how to extract γ from $(typeof(ret))")
    end
end

"""
    match_states(hmm_ref, hmm_other, obs_seq, seq_ends)

Run forward–backward for both HMMs, then find the permutation `σ` mapping
`hmm_other`'s state labels onto `hmm_ref`'s: state `k` in the reference
corresponds to state `σ[k]` in the other model. Chosen as the permutation
maximizing ∑_t ∑_k γ_ref[k,t] · γ_other[σ[k],t].
"""
function match_states(hmm_ref, hmm_other, obs_seq, seq_ends)
    γ_ref = _fb_gamma(hmm_ref, obs_seq, seq_ends)
    γ_other = _fb_gamma(hmm_other, obs_seq, seq_ends)
    K = size(γ_ref, 1)
    O = γ_ref * γ_other'
    best_perm = collect(1:K)
    best_score = -Inf
    for σ in all_permutations(K)
        s = sum(O[k, σ[k]] for k in 1:K)
        if s > best_score
            best_score = s
            best_perm = σ
        end
    end
    return best_perm, O
end

# Parameter extraction

const DDM_PARAMS = (:B, :v, :a₀, :τ)

function params_table(hmm)
    K = length(hmm.dists)
    df = DataFrame(; state=collect(1:K))
    for p in DDM_PARAMS
        df[!, p] = [getfield(hmm.dists[k], p) for k in 1:K]
    end
    return df
end

between_state_std(xs) = std(xs; corrected=false)
between_state_range(xs) = maximum(xs) - minimum(xs)
between_state_cv(xs) = between_state_std(xs) / (mean(xs) + eps())

# Main: compare tied-v vs full for each winning rat

results_dir = joinpath("results", "ddm_hmm_constrained")
out_dir = joinpath(results_dir, "v_tied_compensation")
isdir(out_dir) || mkpath(out_dir)

winners = CSV.read(joinpath(results_dir, "bic_winners.csv"), DataFrame)
v_rats = winners[winners.winning_tied .== "v", :rat]
@info "Rats where v-tied won: $(v_rats)"

# Long-form table: one row per (rat, state, parameter) with matched full & tied values
per_state_rows = DataFrame(;
    rat=String[],
    matched_state=Int[],    # state index in the matched ordering
    parameter=Symbol[],
    full_value=Float64[],
    tied_value=Float64[],
    abs_delta=Float64[],
    rel_delta=Float64[],   # (tied - full) / |full|
)

# Order-invariant spread table: one row per (rat, parameter)
spread_rows = DataFrame(;
    rat=String[],
    parameter=Symbol[],
    full_std=Float64[],
    tied_std=Float64[],
    full_range=Float64[],
    tied_range=Float64[],
    full_cv=Float64[],
    tied_cv=Float64[],
    range_ratio=Float64[],   # tied / full; > 1 → spread increased
    std_ratio=Float64[],
)

# Per-rat ranking of which parameter moved most (by mean abs relative delta)
rank_rows = DataFrame(;
    rat=String[],
    parameter=Symbol[],
    mean_abs_rel_delta=Float64[],
    max_abs_rel_delta=Float64[],
)

for rat in v_rats
    @info "Analyzing rat $rat"

    full_path = joinpath(results_dir, "$(rat)_K4_tied-full.bson")
    v_path = joinpath(results_dir, "$(rat)_K4_tied-v.bson")
    @load full_path fit
    fit_full = fit
    @load v_path fit
    fit_v = fit

    obs_seq, seq_ends = data_for_rat(rat)

    σ, _ = match_states(fit_full.hmm, fit_v.hmm, obs_seq, seq_ends)

    tbl_full = params_table(fit_full.hmm)
    tbl_v = params_table(fit_v.hmm)
    # Reorder tied-v rows so row k corresponds to full's state k.
    tbl_v_matched = tbl_v[σ, :]
    tbl_v_matched.state = collect(1:nrow(tbl_v_matched))

    # Save raw side-by-side table for this rat.
    per_rat_wide = DataFrame(; state=tbl_full.state)
    for p in DDM_PARAMS
        per_rat_wide[!, Symbol("full_", p)] = tbl_full[!, p]
        per_rat_wide[!, Symbol("tied_", p)] = tbl_v_matched[!, p]
        per_rat_wide[!, Symbol("Δ_", p)] = tbl_v_matched[!, p] .- tbl_full[!, p]
    end
    CSV.write(joinpath(out_dir, "$(rat)_matched_params.csv"), per_rat_wide)

    # Per-state long rows
    for k in 1:nrow(tbl_full)
        for p in DDM_PARAMS
            fv = tbl_full[k, p]
            tv = tbl_v_matched[k, p]
            push!(
                per_state_rows,
                (
                    rat=rat,
                    matched_state=k,
                    parameter=p,
                    full_value=fv,
                    tied_value=tv,
                    abs_delta=tv - fv,
                    rel_delta=(tv - fv) / (abs(fv) + eps()),
                ),
            )
        end
    end

    # Spread rows (order-invariant)
    for p in DDM_PARAMS
        xs_full = tbl_full[!, p]
        xs_v = tbl_v[!, p]     # note: use unmatched here; spread is permutation-invariant
        push!(
            spread_rows,
            (
                rat=rat,
                parameter=p,
                full_std=between_state_std(xs_full),
                tied_std=between_state_std(xs_v),
                full_range=between_state_range(xs_full),
                tied_range=between_state_range(xs_v),
                full_cv=between_state_cv(xs_full),
                tied_cv=between_state_cv(xs_v),
                range_ratio=between_state_range(xs_v) /
                            (between_state_range(xs_full) + eps()),
                std_ratio=between_state_std(xs_v) / (between_state_std(xs_full) + eps()),
            ),
        )
    end

    # Per-parameter ranking for this rat
    for p in DDM_PARAMS
        rs = per_state_rows[
            (per_state_rows.rat .== rat) .& (per_state_rows.parameter .== p), :rel_delta
        ]
        push!(
            rank_rows,
            (
                rat=rat,
                parameter=p,
                mean_abs_rel_delta=mean(abs.(rs)),
                max_abs_rel_delta=maximum(abs.(rs)),
            ),
        )
    end
end

CSV.write(joinpath(out_dir, "per_state_deltas.csv"), per_state_rows)
CSV.write(joinpath(out_dir, "spread_changes.csv"), spread_rows)
CSV.write(joinpath(out_dir, "rat_parameter_ranks.csv"), rank_rows)

# Aggregate summaries

param_summary = combine(groupby(spread_rows, :parameter)) do sub
    (
        mean_std_ratio=mean(sub.std_ratio),
        median_std_ratio=median(sub.std_ratio),
        mean_range_ratio=mean(sub.range_ratio),
        median_range_ratio=median(sub.range_ratio),
        n_rats_increased_spread=sum(sub.std_ratio .> 1.0),
        n_rats=nrow(sub),
    )
end
CSV.write(joinpath(out_dir, "parameter_spread_summary.csv"), param_summary)

rank_summary = combine(groupby(rank_rows, :parameter)) do sub
    (
        mean_abs_rel_delta_mean=mean(sub.mean_abs_rel_delta),
        median_abs_rel_delta_mean=median(sub.mean_abs_rel_delta),
        max_abs_rel_delta_mean=mean(sub.max_abs_rel_delta),
    )
end
CSV.write(joinpath(out_dir, "parameter_rank_summary.csv"), rank_summary)

# Which parameter moved most for each rat?
top_mover = combine(groupby(rank_rows, :rat)) do sub
    i = argmax(sub.mean_abs_rel_delta)
    (top_parameter=sub.parameter[i], mean_abs_rel_delta=sub.mean_abs_rel_delta[i])
end
CSV.write(joinpath(out_dir, "top_mover_per_rat.csv"), top_mover)

@info "Per-parameter spread summary (across v-winning rats):"
show(param_summary; allrows=true, allcols=true);
println()
@info "Top mover per rat:"
show(top_mover; allrows=true, allcols=true);
println()

# Plots

param_order = [:B, :v, :a₀, :τ]
param_labels = Dict(
    :B => "B (boundary)",
    :v => "v (drift, tied)",
    :a₀ => "a₀ (bias)",
    :τ => "τ (non-decision)",
)

# ---- Plot 1: distribution of std_ratio per parameter across rats ----
rank_tbl_param = [findfirst(==(p), param_order) for p in spread_rows.parameter]

p_std = boxplot(
    rank_tbl_param,
    spread_rows.std_ratio;
    fillalpha=0.4,
    linecolor=:black,
    legend=false,
    xlabel="DDM parameter",
    ylabel="between-state std (tied-v / full)",
    title="Compensation in v-tied model (K=4): between-state spread ratio",
    xticks=(1:length(param_order), [param_labels[p] for p in param_order]),
    size=(750, 500),
)
dotplot!(
    p_std,
    rank_tbl_param,
    spread_rows.std_ratio;
    marker=(:circle, 5, 0.7, stroke(0)),
    color=:steelblue,
)
hline!(p_std, [1.0]; color=:black, linestyle=:dash, linewidth=1)
savefig_both(p_std, joinpath(out_dir, "between_state_std_ratio"))

# ---- Plot 2: mean absolute relative delta per parameter across rats ----
rank_tbl_param2 = [findfirst(==(p), param_order) for p in rank_rows.parameter]
p_delta = boxplot(
    rank_tbl_param2,
    rank_rows.mean_abs_rel_delta;
    fillalpha=0.4,
    linecolor=:black,
    legend=false,
    xlabel="DDM parameter",
    ylabel="mean |Δ| / |full value| (per-state, matched)",
    title="Per-state parameter change from full → v-tied",
    xticks=(1:length(param_order), [param_labels[p] for p in param_order]),
    size=(750, 500),
)
dotplot!(
    p_delta,
    rank_tbl_param2,
    rank_rows.mean_abs_rel_delta;
    marker=(:circle, 5, 0.7, stroke(0)),
    color=:indianred,
)
savefig_both(p_delta, joinpath(out_dir, "mean_abs_rel_delta"))

# ---- Plot 3: per-rat per-state scatter: full vs tied for each parameter ----
for p in param_order
    sub = per_state_rows[per_state_rows.parameter .== p, :]
    isempty(sub) && continue
    lo = min(minimum(sub.full_value), minimum(sub.tied_value))
    hi = max(maximum(sub.full_value), maximum(sub.tied_value))
    pad = 0.05 * (hi - lo + eps())
    p_sc = scatter(
        sub.full_value,
        sub.tied_value;
        group=sub.rat,
        xlabel="full model: $(param_labels[p])",
        ylabel="v-tied model: $(param_labels[p])",
        title="$(param_labels[p]) — full vs v-tied (matched states)",
        legend=:outerright,
        markersize=5,
        markerstrokecolor=:white,
        size=(800, 500),
        xlim=(lo - pad, hi + pad),
        ylim=(lo - pad, hi + pad),
    )
    plot!(
        p_sc,
        [lo - pad, hi + pad],
        [lo - pad, hi + pad];
        color=:black,
        linestyle=:dash,
        label="",
    )
    savefig_both(p_sc, joinpath(out_dir, "scatter_full_vs_tied_$(p)"))
end

@info "All outputs written to $out_dir"
