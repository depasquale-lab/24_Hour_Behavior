#=
Reviewer figure: the latent states predict trial initiation, which the model
never saw.

The DDM-HMM is conditioned on reaction time, choice and correct side. How long
the animal waits before starting the next trial, how many trials it produces per
minute, and whether it is about to stop working enter neither the likelihood nor
the state sequence. Every panel here is therefore a held-out prediction of the
fitted posterior.

  A  P(state | ITI): mean posterior in each within-animal decile of the wait
     preceding the trial
  B  posterior-weighted inter-trial interval by state rank
  C  posterior-weighted local trial production rate by state rank
  D  posterior-weighted P(break > 5 min before the next trial) by state rank
  E  engagement index E[rank] over the last ten trials of a work bout, where
     bouts are delimited by breaks longer than five minutes
  F  SPECIFICITY CONTROL, not an effect-size panel. Per-animal Spearman
     correlation between the posterior engagement index E[rank] and log ITI /
     trial rate, each corrected by that animal's own within-session
     circular-shift null.

STATISTICS. The ANIMAL is the unit of replication everywhere, so no p value in
this figure can be inflated by trial count. Each panel gets two numbers:

  a trend test   the per-animal Spearman correlation between state rank and the
                 behaviour across the four states (B, C, D), or between position
                 in the bout and E[rank] (E), tested against zero over animals.
                 This is what licenses reading the panels as an ordered trend.
                 A strictly monotone rank 1 < 2 < 3 < 4 in the raw values holds
                 in only 1/18 animals and is NOT claimed; the rank correlation
                 is.
  a contrast     the rank 1 -> rank 4 fold change (B, C, D) or the change over
                 the last ten trials of a bout (E), with a 95% CI, so the effect
                 is reported in units a reader can interpret.

Both are computed with a one-sample t test on the per-animal values (which gives
the CI) and cross-checked with an exact Wilcoxon signed-rank test (which assumes
nothing about the distribution). We previously used a sign test; it is valid but
discards magnitude, gives no interval, and floors at p = 7.6e-6 at n = 18.

Panel F is deliberately NOT annotated with a tally of animals passing their own
circular-shift permutation test. At 5k-94k trials per rat that test is trivially
passed: 16/18 animals reach p < 0.05, including animals with rho = 0.004. The
shift null is retained for the one job it is actually good at, removing the bias
that the shared autocorrelation of posteriors and ITIs induces in rho, and each
animal's rho is corrected by its own null mean before the animal-level test. The
correction turns out to be small (mean rho 0.115 -> 0.107), which is itself the
reassuring result. A single-trial rank correlation against an ITI this variable
is floor-limited by trial noise, so a small rho here is not a small effect: rat
1054 has rho = +0.004 while its ITI more than doubles from its best to its worst
state. The magnitude of the effect is in B, C and D, in interpretable units.

The hour-of-day control that used to occupy a panel is now printed for the text:
it is redundant with the circular-shift null in F, and a scatter against the
identity line is a poor encoding of a paired change.

Inputs are the four tables written by StateEngagementITI.jl.
=#

using Pkg
Pkg.activate("notebooks")

using CSV
using DataFrames
using Statistics
using StatsBase
using Distributions
using Printf
using Random
using Plots

gr()
Random.seed!(20260915)   # jitter in panel F

# Set a standard font family so text stays editable in Illustrator after SVG export.
const FONT_FAMILY = "Helvetica"
default(;
    fontfamily=FONT_FAMILY,
    titlefontfamily=FONT_FAMILY,
    guidefontfamily=FONT_FAMILY,
    tickfontfamily=FONT_FAMILY,
    legendfontfamily=FONT_FAMILY,
    guidefontsize=9,
    tickfontsize=8,
    titlefontsize=10,
    legendfontsize=7,
    grid=false,
    framestyle=:axes,
)

"savefig both .png and .svg alongside each other."
function savefig_both(p, stem::AbstractString)
    savefig(p, stem * ".png")
    return savefig(p, stem * ".svg")
end

results_dir = joinpath("results", "final_ddmhmms")
summ = CSV.read(joinpath(results_dir, "state_engagement_summary.csv"), DataFrame)
dec = CSV.read(joinpath(results_dir, "state_engagement_deciles.csv"), DataFrame)
stats = CSV.read(joinpath(results_dir, "state_engagement_stats.csv"), DataFrame)
bout = CSV.read(joinpath(results_dir, "state_bout_profile.csv"), DataFrame)

const K = 4
const NRAT = length(unique(summ.rat))
const BOUT_POS = 10          # trials profiled at each edge of a work bout
rank_colors = palette(:viridis, K)

sem(v) = std(v) / sqrt(length(v))

# --- animal-level tests -------------------------------------------------------
# Every test below takes one value per animal. The per-animal permutation tests
# in the stats table use the trial as the unit, so at 5k-94k trials per rat they
# reach significance on effects of no interest; nothing here can.

"One-sample t test against `μ0`. Returns the mean, its 95% CI, t, df and p."
function ttest1(v::AbstractVector{<:Real}; μ0::Real=0.0)
    n = length(v)
    m = mean(v)
    se = std(v) / sqrt(n)
    df = n - 1
    t = (m - μ0) / se
    p = 2 * ccdf(TDist(df), abs(t))
    q = quantile(TDist(df), 0.975)
    return (; m, se, t, df, p, lo=m - q * se, hi=m + q * se, n)
end

"""
Exact two-sided Wilcoxon signed-rank test against zero.

Zeros are dropped. Ranks are doubled so that mid-ranks from ties stay integral,
which lets the null be enumerated exactly by convolution over the 2^n sign
assignments rather than approximated by a normal.
"""
function signrank(v::AbstractVector{<:Real})
    d = filter(!iszero, v)
    n = length(d)
    n == 0 && return (; W=0.0, n=0, p=1.0)
    r2 = round.(Int, 2 .* tiedrank(abs.(d)))
    tot = sum(r2)
    W = sum(r2[i] for i in 1:n if d[i] > 0; init=0)
    counts = zeros(Float64, tot + 1)
    counts[1] = 1.0
    for rr in r2
        nc = zeros(Float64, tot + 1)
        for s in 0:tot
            c = counts[s + 1]
            c == 0 && continue
            nc[s + 1] += c
            s + rr <= tot && (nc[s + rr + 1] += c)
        end
        counts = nc
    end
    lo, hi = minmax(W, tot - W)
    p = (sum(counts[1:(lo + 1)]) + sum(counts[(hi + 1):end])) / 2.0^n
    return (; W=W / 2, n, p=min(1.0, p))
end

"Both animal-level tests on one vector, formatted for the caption."
function report(label::AbstractString, v::AbstractVector{<:Real};
                dir::Symbol=:pos, exp_scale::Bool=false)
    t = ttest1(v)
    w = signrank(v)
    k = count(dir === :pos ? (>(0)) : (<(0)), v)
    if exp_scale
        @printf("  %-34s %.2fx  95%% CI [%.2f, %.2f]  t(%d) = %.2f, p = %.1e  |  signed-rank p = %.1e  |  %d/%d\n",
                label, exp(t.m), exp(t.lo), exp(t.hi), t.df, t.t, t.p, w.p, k, t.n)
    else
        @printf("  %-34s %+.3f  95%% CI [%+.3f, %+.3f]  t(%d) = %.2f, p = %.1e  |  signed-rank p = %.1e  |  %d/%d\n",
                label, t.m, t.lo, t.hi, t.df, t.t, t.p, w.p, k, t.n)
    end
    return t
end

"Per-animal Spearman correlation between state rank and `col`, one value per rat."
function rank_trend(col::Symbol)
    return [corspearman(Float64.(sort(summ[summ.rat .== rat, :], :acc_rank).acc_rank),
                        Float64.(sort(summ[summ.rat .== rat, :], :acc_rank)[!, col]))
            for rat in unique(summ.rat)]
end

"Per-animal log fold change from rank 1 to rank K in `col`."
function rank_contrast(col::Symbol)
    return [let s = sort(summ[summ.rat .== rat, :], :acc_rank)[!, col]
                log(s[end] / s[1])
            end for rat in unique(summ.rat)]
end

"Mean ± SEM across animals of `col`, grouped by `by`, one row per group."
function across_rats(df::DataFrame, by, col::Symbol)
    g = combine(groupby(df, by), col => mean => :y, col => sem => :e, nrow => :n)
    return sort(g, by isa Symbol ? by : by[1])
end

# Panel A: P(state | ITI decile)

pA = plot(;
    xlabel="preceding inter-trial interval (s)",
    ylabel="mean posterior P(state)",
    xscale=:log10,
    ylims=(0.04, 0.62),
    yticks=([0.1, 0.2, 0.3, 0.4, 0.5], ["0.1", "0.2", "0.3", "0.4", "0.5"]),
    legend=:topright,
    legend_columns=2,
    foreground_color_legend=nothing,
    background_color_legend=nothing,
    title="A   P(state) vs. the wait before the trial",
    titlelocation=:left,
)
for rank in 1:K
    sub = dec[dec.acc_rank .== rank, :]
    g = combine(groupby(sub, :decile), :iti_pre => mean => :x,
                :posterior => mean => :y, :posterior => sem => :e)
    sort!(g, :decile)
    plot!(pA, g.x, g.y; yerror=g.e, color=rank_colors[rank], linewidth=2,
          marker=(:circle, 3.5, stroke(0)), markercolor=rank_colors[rank],
          markerstrokecolor=rank_colors[rank], label="rank $rank")
end

"""
Format the animal-level trend statistic for an on-panel annotation.

The number drawn on B-E is the MEAN over animals of the per-animal Spearman
correlation, with the p value of the one-sample t test on those `NRAT` values.
It is the trend test described in the header: it uses the animal as the unit of
replication and says nothing about any individual pair of ranks.
"""
function trend_label(v::AbstractVector{<:Real})
    t = ttest1(v)
    e = floor(Int, log10(t.p))
    mant = t.p / 10.0^e
    return @sprintf("mean ρ = %+.2f\np = %.1f×10%s", t.m, mant, superscript(e))
end

"Render an integer exponent with Unicode superscripts, so GR needs no markup."
function superscript(n::Integer)
    digits = ('⁰', '¹', '²', '³', '⁴',
              '⁵', '⁶', '⁷', '⁸', '⁹')
    body = join(digits[d + 1] for d in reverse!(Base.digits(abs(n))))
    return (n < 0 ? "⁻" : "") * body
end

"Draw `trend_label` in a corner of `p` given as fractions of the plotted range."
function annotate_trend!(p, v; xf=0.04, yf=0.96, halign=:left, valign=:top)
    xl, yl = Plots.xlims(p), Plots.ylims(p)
    ylog = p[1][:yaxis][:scale] === :log10
    x = xl[1] + xf * (xl[2] - xl[1])
    y = ylog ? exp10(log10(yl[1]) + yf * (log10(yl[2]) - log10(yl[1]))) :
        yl[1] + yf * (yl[2] - yl[1])
    annotate!(p, x, y, text(trend_label(v), 7, :gray25, halign, valign))
    return p
end

"""
Per-animal lines in grey with the across-animal average on top.

`geo = true` draws the geometric mean with error bars from the SEM of the logs.
Panels B and D span roughly two orders of magnitude across animals and are drawn
on a log axis, where the geometric mean is the matching centre and the one
consistent with the multiplicative rank 1 -> rank K contrast reported for them.
"""
function rank_panel(col::Symbol; ylabel, title, yscale=:identity, yticks=:auto,
                    ylims=:auto, geo::Bool=false)
    p = plot(;
        xlabel="state rank (1 = most accurate)",
        ylabel=ylabel,
        legend=false,
        xticks=(1:K, string.(1:K)),
        xlims=(0.7, K + 0.3),
        yscale=yscale,
        yticks=yticks,
        ylims=ylims,
        title=title,
        titlelocation=:left,
    )
    for rat in unique(summ.rat)
        sub = sort(summ[summ.rat .== rat, :], :acc_rank)
        plot!(p, sub.acc_rank, sub[!, col]; color=:gray75, linewidth=0.8,
              marker=(:circle, 2, 0.5, stroke(0)), markercolor=:gray65)
    end
    vals = [summ[summ.acc_rank .== k, col] for k in 1:K]
    if geo
        m = [exp(mean(log.(v))) for v in vals]
        elo = m .- [exp(mean(log.(v)) - sem(log.(v))) for v in vals]
        ehi = [exp(mean(log.(v)) + sem(log.(v))) for v in vals] .- m
        err = (elo, ehi)
    else
        m = mean.(vals)
        err = sem.(vals)
    end
    plot!(p, 1:K, m; yerror=err, color=:black, linewidth=2,
          marker=(:circle, 4, stroke(0)), markercolor=:black, markerstrokecolor=:black)
    return p
end

# Panels B, C and D: initiation latency, trial production and the probability
# that the animal is about to stop working, all by state rank.
#
# Each carries its animal-level trend statistic on the axes, so the panel is
# readable without the caption. The rank 1 -> rank K contrast with its CI is
# printed below for the caption. Note that the annotated ρ is the mean of the
# per-animal rank correlations, NOT a correlation computed on pooled trials.

trend_iti = rank_trend(:iti_pre_geo)
trend_rate = rank_trend(:trial_rate)
trend_pause = rank_trend(:p_pause_next)

pB = rank_panel(:iti_pre_geo; ylabel="inter-trial interval (s)", yscale=:log10,
                yticks=([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"]),
                ylims=(0.5, 45), geo=true,
                title="B   waits lengthen in worse states")
annotate_trend!(pB, trend_iti)
pC = rank_panel(:trial_rate; ylabel="trial production (trials / min)",
                ylims=(1.2, 6.4),
                title="C   trial rate falls in worse states")
annotate_trend!(pC, trend_rate; xf=0.96, yf=0.96, halign=:right)
pD = rank_panel(:p_pause_next; ylabel="P(break > 5 min before next trial)",
                yscale=:log10,
                yticks=([0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
                        ["0.002", "0.005", "0.01", "0.02", "0.05", "0.1"]),
                ylims=(0.0012, 0.19), geo=true,
                title="D   worse states precede stopping")
annotate_trend!(pD, trend_pause)

# Panel E: the engagement index over the run-up to a break.
#
# E[rank] = Σ_k rank_k · P(state k) collapses the four posteriors onto one
# interpretable axis (1 = the animal's most accurate state, K = its least), so
# the run-up to a break is one line per animal instead of K. Bout onset is not
# shown: the warm-up there is not supported at the animal level (mean per-animal
# Spearman -0.27, 10/18, p = 0.14) and is reported as a null in the text.

erank = combine(groupby(bout, [:rat, :edge, :pos]),
                [:acc_rank, :posterior] => ((r, p) -> sum(r .* p)) => :erank)
bend = sort(erank[erank.edge .== "end", :], [:rat, :pos])

# Animals differ in baseline E[rank] by more than a whole rank (1.48 to 2.69),
# which is a between-animal offset the paired test never sees. Each animal is
# therefore drawn relative to its own mean over the ten positions, so the panel
# shows the quantity that is actually tested. Centring shifts every animal by a
# constant, so the group trace keeps its shape exactly.
transform!(groupby(bend, :rat), :erank => (v -> v .- mean(v)) => :erank_c)

pE = plot(;
    xlabel="trial position relative to a break > 5 min",
    ylabel="E[rank], within animal   (↑ = worse states)",
    legend=false,
    xticks=([-10, -5, -1], ["−10", "−5", "last"]),
    xlims=(-10.6, -0.4),
    title="E   states worsen into a break",
    titlelocation=:left,
)
hline!(pE, [0.0]; color=:gray85, linewidth=0.6)
for rat in unique(bend.rat)
    sub = sort(bend[bend.rat .== rat, :], :pos)
    plot!(pE, -sub.pos, sub.erank_c; color=:gray75, linewidth=0.8)
end
ge = across_rats(bend, :pos, :erank_c)
plot!(pE, -ge.pos, ge.y; yerror=ge.e, color=:black, linewidth=2,
      marker=(:circle, 3.5, stroke(0)), markercolor=:black, markerstrokecolor=:black)

# Trend here is over position in the bout, not over state rank, and is negated
# so the annotated sign matches the direction drawn: ρ > 0 means E[rank] rises
# as the break is approached.
trend_bout = [-corspearman(Float64.(sort(bend[bend.rat .== r, :], :pos).pos),
                           sort(bend[bend.rat .== r, :], :pos).erank_c)
              for r in unique(bend.rat)]
annotate_trend!(pE, trend_bout)

# Panel F: per-animal correlation of the engagement index with initiation
# behaviour, corrected by each animal's own circular-shift null.

# One dot per animal, no per-animal significance encoding: that test is
# trial-level and at these trial counts it passes on rho = 0.004. The shift null
# is used to debias rho, and the inference is the animal-level t test drawn as
# the mean with its 95% CI.
"Observed per-animal values as dots at `x`, with the across-animal mean ± 95% CI."
function rho_column!(p, x, obs; color)
    jitter = x .+ 0.17 .* randn(length(obs))
    scatter!(p, jitter, obs; markercolor=color, markerstrokewidth=0,
             markersize=3.6, alpha=0.6, label="")
    # The interval sits ON the swarm, at the column centre. It is drawn last and
    # given a white halo so it stays legible where it crosses the dots.
    t = ttest1(obs)
    plot!(p, [x, x], [t.lo, t.hi]; color=:white, linewidth=4.5, label="")
    plot!(p, [x, x], [t.lo, t.hi]; color=:black, linewidth=1.6, label="")
    for y in (t.lo, t.hi)
        plot!(p, [x - 0.10, x + 0.10], [y, y]; color=:black, linewidth=1.6, label="")
    end
    plot!(p, [x - 0.30, x + 0.30], [t.m, t.m]; color=:white, linewidth=5.0, label="")
    plot!(p, [x - 0.30, x + 0.30], [t.m, t.m]; color=:black, linewidth=2.4, label="")
    return t
end

pF = plot(;
    ylabel="Spearman ρ with E[rank], shift-corrected",
    legend=false,
    xlims=(0.42, 2.58),
    ylims=(-0.58, 0.38),
    xticks=([1, 2], ["log ITI", "trial rate"]),
    title="F   control: trial-by-trial alignment",
    titlelocation=:left,
)
hline!(pF, [0.0]; color=:black, linestyle=:dash, linewidth=0.7, label="")
rho_pre_c = stats.rho_pre .- stats.rho_pre_null
rho_rate_c = stats.rho_rate .- stats.rho_rate_null
rho_column!(pF, 1.0, rho_pre_c; color=rank_colors[2])
rho_column!(pF, 2.0, rho_rate_c; color=rank_colors[3])
annotate!(pF, 1.5, -0.55,
          text("dot = one animal;  bar = mean, 95% CI", 7, :gray40, :center, :bottom))

fig = plot(pA, pB, pC, pD, pE, pF; layout=(2, 3), size=(1150, 700),
           left_margin=5Plots.mm, bottom_margin=5Plots.mm, top_margin=5Plots.mm)
savefig_both(fig, joinpath(results_dir, "state_engagement_iti"))

# --- numbers for the response letter -----------------------------------------
# Format: value, 95% CI, one-sample t test, exact Wilcoxon signed-rank, and the
# count of animals in the predicted direction. n = NRAT animals throughout.

println("\n================ animal-level statistics (n = $NRAT rats) ================")

println("\nPANELS B, C, D  trend across the four states")
println("  per-animal Spearman(state rank, behaviour), tested against zero")
report("B  rho(rank, ITI)", rank_trend(:iti_pre_geo); dir=:pos)
report("C  rho(rank, trial rate)", rank_trend(:trial_rate); dir=:neg)
report("D  rho(rank, P(break next))", rank_trend(:p_pause_next); dir=:pos)

println("\nPANELS B, C, D  rank 1 -> rank $K contrast")
report("B  ITI", rank_contrast(:iti_pre_geo); dir=:pos, exp_scale=true)
report("C  trial rate", rank_contrast(:trial_rate); dir=:neg, exp_scale=true)
report("D  P(break > 5 min next)", rank_contrast(:p_pause_next); dir=:pos, exp_scale=true)

println("\nPANEL E  engagement index over the run-up to a break")
let pos1 = [only(bend[(bend.rat .== r) .& (bend.pos .== 1), :erank]) for r in unique(bend.rat)],
    pos10 = [only(bend[(bend.rat .== r) .& (bend.pos .== BOUT_POS), :erank]) for r in unique(bend.rat)],
    tr = [corspearman(Float64.(sort(bend[bend.rat .== r, :], :pos).pos),
                      sort(bend[bend.rat .== r, :], :pos).erank) for r in unique(bend.rat)]

    report("E  rho(trials before break, E[rank])", tr; dir=:neg)
    report("E  E[rank], last trial − 10 before", pos1 .- pos10; dir=:pos)
end
let bs = sort(erank[erank.edge .== "start", :], [:rat, :pos]),
    tr = [corspearman(Float64.(sort(bs[bs.rat .== r, :], :pos).pos),
                      sort(bs[bs.rat .== r, :], :pos).erank) for r in unique(bs.rat)]

    println("  (bout onset, reported as a null:)")
    report("   rho(trials after break, E[rank])", tr; dir=:neg)
end

println("\nPANEL F  trial-by-trial alignment, shift-corrected")
report("F  rho(E[rank], log ITI)", rho_pre_c; dir=:pos)
report("F  rho(E[rank], trial rate)", rho_rate_c; dir=:neg)
@printf("  uncorrected for comparison: log ITI mean %+.3f, trial rate mean %+.3f\n",
        mean(stats.rho_pre), mean(stats.rho_rate))
@printf("  per-animal shift test, NOT reported on the panel: %d/%d and %d/%d animals at p < 0.05;\n",
        count(<(0.05), stats.rho_pre_p), NRAT, count(<(0.05), stats.rho_rate_p), NRAT)
@printf("  smallest |rho| among those: %.3f and %.3f (why the trial-level tally is uninformative)\n",
        minimum(abs.(stats.rho_pre[stats.rho_pre_p .< 0.05])),
        minimum(abs.(stats.rho_rate[stats.rho_rate_p .< 0.05])))

println("\nCIRCADIAN CONTROL  for the text, no longer a panel")
report("  rho within hour of day", stats.rho_pre_withinhour .- stats.rho_pre_withinhour_null;
       dir=:pos)
@printf("  median fraction of the raw rho retained: %.2f\n",
        median(stats.rho_pre_withinhour[stats.rho_pre .> 0.002] ./
               stats.rho_pre[stats.rho_pre .> 0.002]))
report("  change from removing hour means", stats.rho_pre_withinhour .- stats.rho_pre; dir=:neg)

println("\nMONOTONICITY  (what the panels may and may not claim)")
let strict = count(unique(summ.rat)) do rat
        v = sort(summ[summ.rat .== rat, :], :acc_rank).iti_pre_geo
        all(v[i] < v[i + 1] for i in 1:(K - 1))
    end
    @printf("  strictly monotone ITI across all %d ranks: %d/%d animals\n", K, strict, NRAT)
    println("  -> the figure claims an ordered trend (rank correlation), not a strict ordering")
end

println("\nPOSTERIOR-WEIGHTED BEHAVIOUR BY STATE RANK (mean over $NRAT rats)")
pooled = combine(groupby(summ, :acc_rank),
                 :iti_pre_geo => mean => :iti, :trial_rate => mean => :rate,
                 :p_pause_next => mean => :p_pause)
for r in eachrow(pooled)
    @printf("  rank %d: ITI %.2f s, %.2f trials/min, P(break > 5 min next) %.3f\n",
            r.acc_rank, r.iti, r.rate, r.p_pause)
end

println("\nPER-ANIMAL EFFECT SIZE, rank 1 -> rank $K")
fold = combine(groupby(summ, :rat)) do g
    srt = sort(g, :acc_rank)
    (iti_fold=srt.iti_pre_geo[end] / srt.iti_pre_geo[1],
     rate_fold=srt.trial_rate[end] / srt.trial_rate[1])
end
for r in eachrow(sort(fold, :iti_fold))
    rho = only(stats[stats.rat .== r.rat, :rho_pre])
    @printf("  %-8s ITI x%.2f   trial rate x%.2f   (rho %+.3f)\n",
            r.rat, r.iti_fold, r.rate_fold, rho)
end
println()
