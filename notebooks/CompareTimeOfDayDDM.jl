#=
Time-of-day DDM vs DDM-HMM on the 24 hr animals (reviewer comparison).

Inputs:
  results/tod_ddm/tod_ddm_summary.csv                       (FitTimeOfDayDDM.jl merge)
  results/ddm_hmm_state_sweep/cv_24hr/cv_state_sweep_summary.csv (GROUP=24hr MergeStateSweepCV.jl)
  results/bic_summary.csv                                   (all-data K=4 DDM-HMM fits)

Held-out logL is pooled per trial over folds. A (rat, K) cell missing folds is
scored on the folds it has, and every comparison against it uses the same folds.
The time-of-day model's best H is chosen from 0..3, so it can fall back to the DDM.

Outputs: <TOD_DIR>/tod_vs_hmm_cv.csv, <TOD_DIR>/tod_vs_hmm_bic.csv
=#

using Pkg
Pkg.activate("notebooks")

using CSV, DataFrames, Statistics, Printf

# VARY_TAU=1 compares the variant where τ also follows time of day.
const TOD_DIR = joinpath("results", get(ENV, "VARY_TAU", "0") == "1" ? "tod_ddm_vary_tau" : "tod_ddm")

tod = CSV.read(joinpath(TOD_DIR, "tod_ddm_summary.csv"), DataFrame; types=Dict(:rat => String))
hmm = CSV.read(joinpath("results", "ddm_hmm_state_sweep", "cv_24hr", "cv_state_sweep_summary.csv"), DataFrame;
               types=Dict(:rat => String))
bic = CSV.read(joinpath("results", "bic_summary.csv"), DataFrame; types=Dict(:rat => String))

cv = tod[tod.fold .> 0, :]

"Per-trial held-out logL pooled over `folds`."
pooled(d, folds) = let s = d[in(folds).(d.fold), :]
    sum(s.test_logL) / sum(s.n_test_trials)
end

rows = DataFrame()
for rat in sort(unique(hmm.rat))
    t, h = cv[cv.rat .== rat, :], hmm[hmm.rat .== rat, :]
    h4 = h[h.K .== 4, :]
    folds4 = sort(h4.fold)

    tod_H = [(H, pooled(t[t.H .== H, :], folds4)) for H in 0:3]
    H_best, tod_ll = tod_H[argmax(last.(tod_H))]
    ddm_ll = pooled(t[t.H .== 0, :], folds4)
    hmm4_ll = pooled(h4, folds4)

    # Best K among cells with all five folds.
    full = [(K, pooled(h[h.K .== K, :], 1:5)) for K in 1:5 if count(==(K), h.K) == 5]
    K_best, _ = full[argmax(last.(full))]

    fold_wins = count(f -> pooled(h4, [f]) > pooled(t[t.H .== H_best, :], [f]), folds4)

    push!(rows, (
        rat=rat, n_test_trials=sum(t.n_test_trials[(t.H .== 0) .& in(folds4).(t.fold)]),
        n_folds=length(folds4), ddm=ddm_ll, tod_best_H=H_best, tod=tod_ll, hmm_K4=hmm4_ll,
        hmm_best_K=K_best, tod_gain=tod_ll - ddm_ll, hmm4_gain=hmm4_ll - ddm_ll,
        hmm4_beats_tod_folds=fold_wins,
    ))
end
CSV.write(joinpath(TOD_DIR, "tod_vs_hmm_cv.csv"), rows)

println("Held-out logL per trial (nats), 24 hr rats")
show(rows; allrows=true, allcols=true)
println()
@printf("\nmedian gain over DDM: time-of-day %.4f | DDM-HMM K=4 %.4f\n",
        median(rows.tod_gain), median(rows.hmm4_gain))
@printf("DDM-HMM K=4 beats best time-of-day DDM: %d/%d rats, %d/%d folds\n",
        count(rows.hmm4_gain .> rows.tod_gain), nrow(rows),
        sum(rows.hmm4_beats_tod_folds), sum(rows.n_folds))
@printf("time-of-day DDM worse than DDM on held-out days at every H>0: %d rats\n", count(rows.tod_best_H .== 0))

# BIC on all data: best H vs best tied K=4 variant.
brows = DataFrame()
for rat in sort(unique(tod.rat))
    t = tod[(tod.rat .== rat) .& (tod.fold .== 0), :]
    b = bic[bic.rat .== rat, :]
    i, j = argmin(t.bic), argmin(b.bic)
    push!(brows, (rat=rat, bic_ddm=t.bic[t.H .== 0][1], tod_best_H=t.H[i], bic_tod=t.bic[i],
                  hmm_tied=b.tied[j], bic_hmm=b.bic[j], Δbic_hmm_minus_tod=b.bic[j] - t.bic[i]))
end
CSV.write(joinpath(TOD_DIR, "tod_vs_hmm_bic.csv"), brows)

println("\nBIC on all data")
show(brows; allrows=true, allcols=true)
println()
@printf("\nDDM-HMM lower BIC than time-of-day DDM: %d/%d rats (median ΔBIC %.0f)\n",
        count(brows.Δbic_hmm_minus_tod .< 0), nrow(brows), median(brows.Δbic_hmm_minus_tod))
@printf("time-of-day DDM lower BIC than DDM: %d/%d rats\n",
        count(brows.bic_tod .< brows.bic_ddm), nrow(brows))
