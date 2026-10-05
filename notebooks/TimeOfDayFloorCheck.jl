#=
Held-out logL per trial for the DDM, time-of-day DDM (VARY_TAU=1 fits) and K=4 DDM-HMM
(one-step-ahead predictive), with and without trials that hit the WFPT density
floor (RT <= tau) under the DDM or time-of-day DDM. Run after CompareTimeOfDayDDM.jl.

Outputs: results/tod_ddm_vary_tau/floor_check{,_by_fold}.csv
=#

using Pkg
Pkg.activate("notebooks")

using DriftDiffusionModels, HiddenMarkovModels, BSON, CSV, DataFrames, Dates, Printf, Statistics
const FLOOR = log(1e-16)
df = CSV.read(joinpath("data","processed_rat_data.csv.gz"), DataFrame; types=Dict(:name=>String))
df = df[df.daily .== "24 hr", :]
replace!(df.choose_right, 0 => -1)
df.s = [cs == "right" ? 1 : -1 for cs in df.correct_side]
hr(dt) = (p = split(dt); length(p) < 2 ? 0.0 : sum(parse.(Float64, split(p[2], ':')) .* [1, 1/60, 1/3600]))
cmp = CSV.read("results/tod_ddm_vary_tau/tod_vs_hmm_cv.csv", DataFrame; types=Dict(:rat=>String))
lse(v) = (m = maximum(v); m + log(sum(exp.(v .- m))))

"Per-trial one-step-ahead predictive logL under an HMM, per session."
function hmm_pertrial(hmm, sessions)
    lA = log.(hmm.trans); out = Float64[]
    for obs in sessions
        la = log.(hmm.init)
        for (t, o) in enumerate(obs)
            pred = t == 1 ? la : [lse(la .+ lA[:, j]) for j in axes(lA, 2)]
            e = [logdensityof(d, o) for d in hmm.dists]
            joint = pred .+ e
            l = lse(joint); push!(out, l); la = joint .- l
        end
    end
    out
end

rows = DataFrame()
for r in eachrow(cmp)
    rat, H = r.rat, r.tod_best_H
    s = df[df.name .== rat, :]
    dates = [Date(split(dt)[1]) for dt in s.trial_datetime]
    ud = sort(unique(dates)); sess = [findall(dates .== d) for d in ud]
    n = length(sess); fs = div(n, 5)
    fits = BSON.load("results/tod_ddm_vary_tau/per_rat/$(rat)_tod_fits.bson")[:fits]
    for fold in 1:5
        f = "results/ddm_hmm_state_sweep/cv_24hr/$(rat)_K4_fold$(fold)_24hr_cv.bson"
        isfile(f) || continue
        b = BSON.load(f); hmm = b[:hmm]
        te = ((fold-1)*fs+1):(fold == 5 ? n : fold*fs)
        idx = sess[te]
        hs = hmm_pertrial(hmm, [[DDMResult(s.rt[i], s.choose_right[i], s.s[i]) for i in ix] for ix in idx])
        @assert isapprox(sum(hs), b[:test_ll]; rtol=1e-6) "forward mismatch $rat $fold"
        all_i = reduce(vcat, idx)
        mt, m0 = fits[fold][H], fits[fold][0]
        obsH(i, HH) = RegDDMResult(s.rt[i], s.choose_right[i], s.s[i], fourier_basis(hr(String(s.trial_datetime[i])); n_harmonics=HH))
        lt = [logdensityof(RegressionDDM(mt.β, mt.free), obsH(i, H)) for i in all_i]
        l0 = [logdensityof(RegressionDDM(m0.β, m0.free), obsH(i, 0)) for i in all_i]
        fl = (lt .<= FLOOR + 1e-9) .| (l0 .<= FLOOR + 1e-9)
        push!(rows, (rat=rat, fold=fold, n=length(all_i), n_floor=count(fl),
            n_floor_tod=count(lt .<= FLOOR + 1e-9), n_floor_ddm=count(l0 .<= FLOOR + 1e-9),
            ddm=sum(l0), tod=sum(lt), hmm=sum(hs),
            ddm_nf=sum(l0[.!fl]), tod_nf=sum(lt[.!fl]), hmm_nf=sum(hs[.!fl]), n_nf=count(.!fl)))
    end
end
CSV.write("results/tod_ddm_vary_tau/floor_check_by_fold.csv", rows)
g = combine(groupby(rows, :rat), [:n, :n_floor, :n_floor_tod, :n_floor_ddm, :n_nf] .=> sum .=> [:n, :n_floor, :n_floor_tod, :n_floor_ddm, :n_nf],
    [:ddm, :tod, :hmm, :ddm_nf, :tod_nf, :hmm_nf] .=> sum .=> [:ddm, :tod, :hmm, :ddm_nf, :tod_nf, :hmm_nf])
g.tod_gain = (g.tod .- g.ddm) ./ g.n;  g.hmm_gain = (g.hmm .- g.ddm) ./ g.n
g.tod_gain_nf = (g.tod_nf .- g.ddm_nf) ./ g.n_nf; g.hmm_gain_nf = (g.hmm_nf .- g.ddm_nf) ./ g.n_nf
CSV.write("results/tod_ddm_vary_tau/floor_check.csv", g)
println("rat     floored(test)  ToD gain  HMM gain | excl. floored: ToD gain  HMM gain")
for r in eachrow(g)
    @printf("%-7s %4d/%-6d    %+.4f   %+.4f  |              %+.4f   %+.4f\n", r.rat, r.n_floor, r.n, r.tod_gain, r.hmm_gain, r.tod_gain_nf, r.hmm_gain_nf)
end
@printf("\nmedians: all trials ToD %.4f HMM %.4f | excl. floored ToD %.4f HMM %.4f\n", median(g.tod_gain), median(g.hmm_gain), median(g.tod_gain_nf), median(g.hmm_gain_nf))
@printf("HMM > ToD excl. floored: %d/%d rats; floored share of test trials: median %.3f%%, max %.3f%%\n",
    count(g.hmm_gain_nf .> g.tod_gain_nf), nrow(g), 100median(g.n_floor ./ g.n), 100maximum(g.n_floor ./ g.n))
