using Pkg
Pkg.activate("notebooks")

using Random
using DriftDiffusionModels
using Dates
using CSV
using DataFrames
using Statistics

# set random seed 
Random.seed!(67)  # this seed is bussin fr fr on god

# Preprocess data once
data_file = joinpath("data", "processed_rat_data.csv.gz")
rat_df = CSV.read(data_file, DataFrame)
rat_df = rat_df[rat_df.daily .== "24 hr", :]

replace!(rat_df[!, :choose_right], 0 => -1)

mapping = Dict("right" => 1, "left" => -1)
transform!(
    rat_df, :correct_side => ByRow(cs -> get(mapping, cs, missing)) => :correct_side_numeric
)

rat_names = unique(rat_df.name)

function build_all_results_for_rat(rat_df::DataFrame, rat_name::AbstractString)
    rat_sub = rat_df[rat_df.name .== rat_name, :]

    all_results = [
        DDMResult(rt, choice, stim) for (rt, choice, stim) in
        zip(rat_sub.rt, rat_sub.choose_right, rat_sub.correct_side_numeric)
    ]

    return rat_sub, all_results  # rat_sub keeps metadata (rt, choice, date, etc.)
end

function fit_all_rats_and_save(rat_df::DataFrame; n_iter::Int=25, K::Int=100, seed::Int=67)
    rng = MersenneTwister(seed)

    trial_post_df = DataFrame(;
        rat_name=String[],
        trial_idx=Int[],
        rt=Float64[],
        choice=Int[],
        stim=Int[],
        μ_uB=Float64[],
        μ_uτ=Float64[],
        μ_uv=Float64[],
        μ_ua0=Float64[],
        logσ_uB=Float64[],
        logσ_uτ=Float64[],
        logσ_uv=Float64[],
        logσ_ua0=Float64[],
        B_mean=Float64[],
        τ_mean=Float64[],
        v_mean=Float64[],
        a0_mean=Float64[],
    )

    hyper_df = DataFrame(;
        rat_name=String[],
        m_uB=Float64[],
        m_uτ=Float64[],
        m_uv=Float64[],
        m_ua0=Float64[],
        logσ0_uB=Float64[],
        logσ0_uτ=Float64[],
        logσ0_uv=Float64[],
        logσ0_ua0=Float64[],
        B_group_mean=Float64[],
        τ_group_mean=Float64[],
        v_group_mean=Float64[],
        a0_group_mean=Float64[],
    )

    elbo_df = DataFrame(; rat_name=String[], iter=Int[], elbo=Float64[])

    rat_names = unique(rat_df.name)

    for rat_name in rat_names
        println("Fitting pooled eDDM for rat: $rat_name")

        rat_sub, all_results = build_all_results_for_rat(rat_df, rat_name)

        if isempty(all_results)
            @warn "No trials found for rat $rat_name, skipping"
            continue
        end

        # fit one model per rat (all trials pooled)
        hyper, qs, elbo_history = fit_vi_gaussian(
            all_results; n_iter=n_iter, K=K, rng=rng, verbose=false
        )

        # store hyperparameters for this rat
        m = hyper.m
        logσ0 = hyper.logσ

        # transform hyper means to natural space (approximation via transform of mean)
        B_group_mean = exp(m[1])
        τ_group_mean = exp(m[2])
        v_group_mean = exp(m[3])
        a0_group_mean = logistic(m[4])

        push!(
            hyper_df,
            (
                rat_name,
                m[1],
                m[2],
                m[3],
                m[4],
                logσ0[1],
                logσ0[2],
                logσ0[3],
                logσ0[4],
                B_group_mean,
                τ_group_mean,
                v_group_mean,
                a0_group_mean,
            ),
        )

        # store per-trial variational parameters
        @assert length(qs) == nrow(rat_sub)

        for (trial_idx, q) in enumerate(qs)
            μ = q.μ
            logσ = q.logσ

            # transform posterior mean (again, simple transform of mean)
            B_mean = exp(μ[1])
            τ_mean = exp(μ[2])
            v_mean = exp(μ[3])
            a0_mean = logistic(μ[4])

            row = (
                rat_name,
                trial_idx,
                Float64(rat_sub.rt[trial_idx]),
                Int(rat_sub.choose_right[trial_idx]),
                Int(rat_sub.correct_side_numeric[trial_idx]),
                μ[1],
                μ[2],
                μ[3],
                μ[4],
                logσ[1],
                logσ[2],
                logσ[3],
                logσ[4],
                B_mean,
                τ_mean,
                v_mean,
                a0_mean,
            )

            push!(trial_post_df, row)
        end

        # ELBO history for diagnostics
        for (it, elbo_val) in enumerate(elbo_history)
            push!(elbo_df, (rat_name, it, elbo_val))
        end
    end

    # write to CSV
    mkpath("results")

    CSV.write(joinpath("results", "eddm_trial_posteriors_by_rat.csv"), trial_post_df)
    CSV.write(joinpath("results", "eddm_hyperparams_by_rat.csv"), hyper_df)
    CSV.write(joinpath("results", "eddm_elbo_history_by_rat.csv"), elbo_df)

    return trial_post_df, hyper_df, elbo_df
end

trial_post_df, hyper_df, elbo_df = fit_all_rats_and_save(rat_df; n_iter=25, K=100, seed=67)

println("Fitting complete. Results saved to 'results/' directory.")
