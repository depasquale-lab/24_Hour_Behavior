using Pkg
Pkg.activate("notebooks")

using Random
using Distributions
using DriftDiffusionModels
using HiddenMarkovModels
using Dates
using CSV
using DataFrames
using Statistics
using BSON: @save, @load

struct DDMHMMFit
    hmm::PriorHMM
    logL::Float64
    logL_evolution::Vector{Float64}
end

# Unused here; kept for reference.
function load_best_ddmhmm(
    rat::AbstractString, K::Int; results_dir=joinpath("results", "ddm_hmm")
)
    filename = joinpath(results_dir, "$(rat)_K$(K)_best.bson")
    @load filename fit
    return fit::DDMHMMFit
end

const N_ITERS = 10 # how many times to fit the DDM-HMM per animal
const N_STATES = [3, 4] # which hidden states to fit over

# set random seed
Random.seed!(67)  # this seed is bussin fr fr on god

# set data path (assumes you are in the project root)
data_file = joinpath("data", "processed_rat_data.csv.gz")

# Read in and structure data
rat_df = CSV.read(data_file, DataFrame)
rat_df = rat_df[rat_df.daily .== "daily", :] # only keep daily data

# preprocess the data to have numerics
replace!(rat_df[!, :choose_right], 0 => -1)

mapping = Dict("right" => 1, "left" => -1)
DataFrames.transform!(
    rat_df, :correct_side => ByRow(cs -> get(mapping, cs, missing)) => :correct_side_numeric
)

names = String.(unique(rat_df[!, "name"]))

# Build (all_results, seq_ends, results_by_date) for one rat, one sequence per day.
function data_for_ddmhmm(rat_idx::Int)
    rat = names[rat_idx]

    rat_of_interest = rat_df[rat_df.name .== rat, :]
    dates = [Date(split(dt)[1]) for dt in rat_of_interest.trial_datetime]

    unique_dates = sort(unique(dates))
    results_by_date = Vector{Vector{DDMResult}}()

    for date in unique_dates
        day_indices = findall(dates .== date)
        if isempty(day_indices)
            continue
        end

        day_rts = rat_of_interest.rt[day_indices]
        day_outcomes = rat_of_interest.choose_right[day_indices]
        day_stim_side = rat_of_interest.correct_side_numeric[day_indices]

        day_results = [
            DDMResult(rt, choice, stim) for
            (rt, choice, stim) in zip(day_rts, day_outcomes, day_stim_side)
        ]
        push!(results_by_date, day_results)
    end

    seq_ends = cumsum([length(seq) for seq in results_by_date])
    all_results = reduce(vcat, results_by_date)

    return all_results, seq_ends, results_by_date
end

function generate_ddmhmm_initialization(n_states::Int)

    # params we need for ddm-hmm
    init_init = zeros(Float64, n_states)
    trans_init = zeros(Float64, n_states, n_states)
    emissions_init = Vector{DriftDiffusionModel}(undef, n_states)

    # sample initial state distribution from Dirichlet
    init_init .= rand(Dirichlet(fill(1.0, n_states)))

    # sample transition from Dirichlet for each state
    for i in 1:n_states
        dirichlet_vector = ones(n_states)
        dirichlet_vector[i] = 10.0 # bias towards self-transition

        trans_init[i, :] .= rand(Dirichlet(dirichlet_vector))
    end

    # sample DDM parameters for each state
    log_drift = rand(Normal(0.0, 0.5), n_states)
    log_boundary = rand(Normal(log(2), 0.5), n_states)
    bias = rand(Beta(10, 10), n_states)
    non_decision_time = 0.1

    for i in 1:n_states
        emissions_init[i] = DriftDiffusionModel(
            exp(log_drift[i]), exp(log_boundary[i]), bias[i], non_decision_time
        )
    end

    return PriorHMM(init_init, trans_init, emissions_init, 1, 1)
end

"""
    fit_best_ddmhmm_for_rat(rat_idx::Int, n_states::Int;
                            n_inits::Int = max(N_ITERS, 1),
                            atol::Float64 = 1e-3,
                            max_iter::Int = 100,
                            loglikelihood_increasing::Bool = false)

For a given rat and a given number of states `n_states`, run Baum–Welch
from `n_inits` random initializations and return the best-fitting model
according to the final log-likelihood.
"""
function fit_best_ddmhmm_for_rat(
    rat_idx::Int,
    n_states::Int;
    n_inits::Int=max(N_ITERS, 1),
    atol::Float64=1e-3,
    max_iter::Int=100,
    loglikelihood_increasing::Bool=false,
)
    obs_seq, seq_ends, _ = data_for_ddmhmm(rat_idx)

    best_hmm = nothing
    best_logL = -Inf
    best_logL_evolution = Float64[]

    for init_id in 1:n_inits
        prior = generate_ddmhmm_initialization(n_states)

        try
            hmm_est, logL_evolution = HiddenMarkovModels.baum_welch(
                prior,
                obs_seq;
                seq_ends=seq_ends,
                atol=atol,
                max_iterations=max_iter,
                loglikelihood_increasing=loglikelihood_increasing,
            )

            final_ll = last(logL_evolution)

            if final_ll > best_logL
                best_logL = final_ll
                best_hmm = hmm_est
                best_logL_evolution = logL_evolution
            end
        catch e
            @warn "Baum–Welch failed for rat index $rat_idx, K=$n_states, init $init_id" exception=(
                e, catch_backtrace()
            )
        end
    end

    return DDMHMMFit(best_hmm, best_logL, best_logL_evolution)
end

results_dir = joinpath("results", "ddm_hmm")
isdir(results_dir) || mkpath(results_dir)

best_models = Dict{Tuple{String,Int},DDMHMMFit}()

for (rat_idx, rat) in enumerate(names)
    for K in N_STATES
        @info "Fitting DDM-HMM for rat $rat with K = $K using $(max(N_ITERS, 1)) initializations..."

        fit = fit_best_ddmhmm_for_rat(rat_idx, K)
        best_models[(rat, K)] = fit

        filename = joinpath(results_dir, "$(rat)_K$(K)_daily_best.bson")
        @save filename fit rat K

        @info "Saved best model for rat $rat, K = $K to $filename (logL = $(fit.logL))"
    end
end

