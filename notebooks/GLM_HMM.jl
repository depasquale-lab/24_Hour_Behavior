### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ b8595960-ba87-11f0-9aa4-9bbc26e4639c
begin
    using Pkg
    Pkg.activate("/Users/ryansenne/Desktop/Julia_Testing_Env/ssm_dev")

    using StateSpaceDynamics
    using DataFrames
    using CSV
    using Dates
    using LinearAlgebra
    using Plots
end

# ╔═╡ 916432c2-21ef-4d59-8239-a31d9547dc27
md"# Data Preparation"

# ╔═╡ 3facbe41-bd8a-49d7-ad36-f5f9f132c4e2
md"Load the data, pick one rat, split its trials by day (one independent sequence per day)."

# ╔═╡ 52615e7d-58a5-492b-8220-d09d91ffbada
begin
    # read the data
    rat_df = CSV.read("../data/processed_rat_data.csv.gz", DataFrame)
    rat_df = rat_df[rat_df.daily .== "24 hr", :] # only keep 24 hour data

    names = unique(rat_df[!, "name"])

    rat = names[5]

    rat_of_interest = rat_df[rat_df.name .== rat, :]
    dates = [Date(split(dt)[1]) for dt in rat_of_interest.trial_datetime]

    unique_dates = sort(unique(dates))

    resp_var = Vector{Matrix{Float64}}()
    dep_var = Vector{Matrix{Float64}}()

    for date in unique_dates
        day_indices = findall(dates .== date)

        if isempty(day_indices)
            continue
        end

        day_resp = reshape(rat_of_interest.choose_right[day_indices], 1, :)
        day_dep = reshape(rat_of_interest.delta_flashes[day_indices], 1, :)

        push!(resp_var, day_resp)
        push!(dep_var, day_dep)
    end
end

# ╔═╡ 94caf006-c132-446f-a8d7-813479fc151c
begin
    n = length(resp_var)
    train_idxs = Int(0.8 * n)

    train_resp, train_dep = resp_var[1:train_idxs], dep_var[1:train_idxs]
    test_resp, test_dep = resp_var[(train_idxs + 1):end], dep_var[(train_idxs + 1):end]

    init_trans = [0.99 0.01; 0.01 0.99]
    init_dist = [0.5, 0.5]
    B = [
        BernoulliRegressionEmission(1, 1, reshape([0.0, 1.0], :, 1), true, 1.0),
        BernoulliRegressionEmission(1, 1, reshape([0.5, 0.5], :, 1), true, 1.0),
    ]

    hmm = HiddenMarkovModel(init_trans, B, init_dist, 2)
    fit!(hmm, train_resp, train_dep)
end

# ╔═╡ 7b8d68ae-7466-462e-a4b9-ce7060dad67b
hmm.B

# ╔═╡ Cell order:
# ╠═b8595960-ba87-11f0-9aa4-9bbc26e4639c
# ╟─916432c2-21ef-4d59-8239-a31d9547dc27
# ╟─3facbe41-bd8a-49d7-ad36-f5f9f132c4e2
# ╠═52615e7d-58a5-492b-8220-d09d91ffbada
# ╠═94caf006-c132-446f-a8d7-813479fc151c
# ╠═7b8d68ae-7466-462e-a4b9-ce7060dad67b
