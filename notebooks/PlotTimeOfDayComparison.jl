#=
Reviewer figure: time-of-day DDM vs DDM-HMM on the 24 hr animals.

  A  held-out gain over the DDM per rat: time-of-day DDM (τ fixed / τ varying) and
     K = 4 DDM-HMM
  B  DDM-HMM vs time-of-day gain per rat, all held-out trials and excluding trials
     at the WFPT density floor
  C  all-data BIC difference, DDM-HMM minus time-of-day DDM

Inputs come from CompareTimeOfDayDDM.jl (both VARY_TAU settings) and
TimeOfDayFloorCheck.jl. Writes results/tod_ddm_vary_tau/tod_vs_hmm.{png,svg}.
=#

using Pkg
Pkg.activate("notebooks")

using CSV
using DataFrames
using Plots
using Printf
using Statistics

gr()

# Standard font family so text stays editable in Illustrator after SVG export.
const FONT_FAMILY = "Helvetica"
default(;
    fontfamily=FONT_FAMILY,
    titlefontfamily=FONT_FAMILY,
    guidefontfamily=FONT_FAMILY,
    tickfontfamily=FONT_FAMILY,
    legendfontfamily=FONT_FAMILY,
    grid=false,
    framestyle=:axes,
    tickfontsize=8,
    guidefontsize=9,
    titlefontsize=10,
    titlelocation=:left,
    foreground_color_axis=:grey40,
    foreground_color_border=:grey40,
)

"savefig both .png and .svg alongside each other."
function savefig_both(p, stem::AbstractString)
    savefig(p, stem * ".png")
    return savefig(p, stem * ".svg")
end

const C_HMM = colorant"#2a78d6"
const C_TOD = colorant"#eb6834"
const C_RAT = colorant"#c9c8c3"

dir = joinpath("results", "tod_ddm_vary_tau")
rd(f) = CSV.read(f, DataFrame; types=Dict(:rat => String))
fixed = rd(joinpath("results", "tod_ddm", "tod_vs_hmm_cv.csv"))
vary = rd(joinpath(dir, "tod_vs_hmm_cv.csv"))
bic = rd(joinpath(dir, "tod_vs_hmm_bic.csv"))
flr = rd(joinpath(dir, "floor_check.csv"))

d = innerjoin(select(fixed, :rat, :tod_gain => :tod_fixed),
              select(vary, :rat, :tod_gain => :tod, :hmm4_gain => :hmm), on=:rat)
d = innerjoin(d, select(flr, :rat, :tod_gain_nf, :hmm_gain_nf), on=:rat)
n = nrow(d)

# A: paired per-rat gains
xs = [1, 2, 3]
pA = plot(; xticks=(xs, ["Time of day\n(τ fixed)", "Time of day\n(τ varies)", "DDM-HMM\n(K = 4)"]),
          xlims=(0.5, 3.5), ylabel="held-out gain over DDM\n(nats / trial)",
          title="A", legend=false, bottom_margin=6Plots.mm)
hline!(pA, [0]; color=:grey60, ls=:dash, lw=1)
annotate!(pA, 3.45, 0.012, text("DDM", 7, :grey45, :right))
for r in eachrow(d)
    plot!(pA, xs, [r.tod_fixed, r.tod, r.hmm]; color=C_RAT, lw=1)
end
scatter!(pA, fill(1, n), d.tod_fixed; color=:white, markerstrokecolor=C_TOD, ms=4.5, msw=1.5)
scatter!(pA, fill(2, n), d.tod; color=C_TOD, markerstrokecolor=:white, ms=4.5, msw=1)
scatter!(pA, fill(3, n), d.hmm; color=C_HMM, markerstrokecolor=:white, ms=4.5, msw=1)
for (x, v) in zip(xs, (d.tod_fixed, d.tod, d.hmm))
    m = median(v)
    plot!(pA, [x - 0.22, x + 0.22], [m, m]; color=:black, lw=2)
    annotate!(pA, x + 0.26, m, text(@sprintf("%.3f", m), 7, :black, :left))
end

# B: per-rat HMM vs time-of-day gain, with and without floored trials
lim = (0, 0.5)
pB = plot(; xlims=lim, ylims=lim, aspect_ratio=:equal, title="B",
          xlabel="time-of-day gain (nats / trial)", ylabel="DDM-HMM gain (nats / trial)",
          legend=:bottomright, legendfontsize=7, foreground_color_legend=nothing)
plot!(pB, [0, 0.5], [0, 0.5]; color=:grey60, ls=:dash, lw=1, label="")
annotate!(pB, 0.47, 0.44, text("equal", 7, :grey45, :right, rotation=45))
for r in eachrow(d)
    plot!(pB, [r.tod, r.tod_gain_nf], [r.hmm, r.hmm_gain_nf]; color=C_RAT, lw=1, label="")
end
scatter!(pB, d.tod, d.hmm; color=C_HMM, markerstrokecolor=:white, ms=4.5, msw=1,
         label="all held-out trials")
scatter!(pB, d.tod_gain_nf, d.hmm_gain_nf; color=:white, markerstrokecolor=C_HMM, ms=4.5, msw=1.5,
         label="excluding floored trials")
above = count(d.hmm .> d.tod), count(d.hmm_gain_nf .> d.tod_gain_nf)
annotate!(pB, 0.19, 0.16, text("DDM-HMM better in $(above[1])/$n rats\n($(above[2])/$n excluding floored)",
                               7, :grey25, :left, :top))

# C: all-data BIC difference, sorted
b = sort(bic, :Δbic_hmm_minus_tod; rev=true)
yb = 1:nrow(b)
pC = plot(; yticks=(yb, b.rat), ylims=(0.3, nrow(b) + 0.7), title="C",
          xlabel="ΔBIC, DDM-HMM − time of day (thousands)\n< 0 favors DDM-HMM", legend=false, ytickfontsize=7)
vline!(pC, [0]; color=:grey60, ls=:dash, lw=1)
for (y, v) in zip(yb, b.Δbic_hmm_minus_tod)
    plot!(pC, [0, v / 1e3], [y, y]; color=C_RAT, lw=1.5)
end
scatter!(pC, b.Δbic_hmm_minus_tod ./ 1e3, yb; color=C_HMM, markerstrokecolor=:white, ms=4.5, msw=1)

p = plot(pA, pB, pC; layout=grid(1, 3; widths=[0.36, 0.34, 0.30]), size=(1150, 400),
         left_margin=6Plots.mm, bottom_margin=8Plots.mm, dpi=200)
savefig_both(p, joinpath(dir, "tod_vs_hmm"))
@info "wrote $(joinpath(dir, "tod_vs_hmm")).{png,svg}"
