#=
DDM parameters across the day, DDM-HMM vs time-of-day DDM.
  Row 1  one example rat: HMM hourly mean with 10–90 % range, and the time-of-day curve
  Row 2  all rats, each curve centred on its daily mean (shapes compared, not
         levels), in units of the rat's HMM trial-level SD; mean ± SEM

Reads TimeOfDayParameterCurves.jl outputs. Writes results/tod_ddm_vary_tau/param_curves.{png,svg}.
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
    titlefontsize=9,
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
const DARK_SPAN = (12.0, 24.0)
const FEEDING_SPAN = (6.5, 8.5)   # fed 14:00–16:00, as in MiscFigures.jl
const PARAMS = ["B", "v", "a₀", "τ"]
const LABELS = Dict("B" => "bound B", "v" => "drift |v|", "a₀" => "start point a0",
                    "τ" => "non-decision time τ (s)")

dir = joinpath("results", "tod_ddm_vary_tau")
rd(f) = CSV.read(f, DataFrame; types=Dict(:rat => String, :param => String))
curves = rd(joinpath(dir, "param_curves_long.csv"))
vars = rd(joinpath(dir, "param_variance_by_hour.csv"))

r_by = combine(groupby(curves, [:rat, :param]), [:hmm_mean, :tod] => cor => :r)
stats = innerjoin(r_by, vars, on=[:rat, :param])

# Example rat: closest to the across-rat medians of r and variance explained.
med = combine(groupby(stats, :param), :r => median => :r_med, :frac_var_hour => median => :f_med)
dist = combine(groupby(innerjoin(stats, med, on=:param), :rat),
               [:r, :r_med, :frac_var_hour, :f_med] =>
               ((r, rm, f, fm) -> sum(abs.(r .- rm)) + sum(abs.(f .- fm) ./ fm)) => :d)
EXAMPLE = dist.rat[argmin(dist.d)]

base_axis(; kw...) = plot(; xlims=(0, 24), xticks=0:6:24, legend=false, kw...)
function shade!(p)
    vspan!(p, collect(DARK_SPAN); color=:grey92, linewidth=0, label="")
    vspan!(p, collect(FEEDING_SPAN); color=:lightyellow, alpha=0.8, linewidth=0, label="")
end

top = map(PARAMS) do prm
    c = sort(curves[(curves.rat .== EXAMPLE) .& (curves.param .== prm), :], :hour)
    p = base_axis(; title=LABELS[prm], xformatter=_ -> "")
    shade!(p)
    plot!(p, c.hour, c.hmm_q90; fillrange=c.hmm_q10, color=C_HMM, fillalpha=0.18, lw=0, label="")
    plot!(p, c.hour, c.hmm_mean; color=C_HMM, lw=2, label="DDM-HMM (hourly mean, 10–90 % of trials)")
    plot!(p, c.hour, c.tod; color=C_TOD, lw=2, label="time-of-day DDM")
    p
end
plot!(top[1]; ylabel="rat $EXAMPLE")
let c = sort(curves[(curves.rat .== EXAMPLE) .& (curves.param .== "B"), :], :hour)
    annotate!(top[1], 0.6, minimum(c.hmm_q10), text("DDM-HMM (band: 10–90% of trials)", 7, :grey25, :left, :bottom))
    annotate!(top[1], 0.6, maximum(c.tod), text("time-of-day DDM", 7, :grey25, :left, :top))
end

bottom = map(PARAMS) do prm
    g = curves[curves.param .== prm, :]
    z = DataFrame()
    for s in groupby(g, :rat)
        sd = vars.hmm_sd[(vars.rat .== s.rat[1]) .& (vars.param .== prm)][1]
        wmean(x) = sum(x .* s.n) / sum(s.n)
        append!(z, DataFrame(hour=s.hour, hmm=(s.hmm_mean .- wmean(s.hmm_mean)) ./ sd,
                             tod=(s.tod .- wmean(s.tod)) ./ sd))
    end
    a = combine(groupby(z, :hour), :hmm => mean => :hmm, :hmm => (x -> std(x) / sqrt(length(x))) => :hmm_se,
                :tod => mean => :tod, :tod => (x -> std(x) / sqrt(length(x))) => :tod_se)
    sort!(a, :hour)
    ymax = 1.15 * max(1.1, maximum(abs.(vcat(a.hmm .+ a.hmm_se, a.hmm .- a.hmm_se, a.tod .+ a.tod_se, a.tod .- a.tod_se))))
    p = base_axis(; ylims=(-ymax, ymax), xlabel="hours from lights on")
    shade!(p)
    hline!(p, [-1, 1]; color=:grey60, ls=:dash, lw=1)
    hline!(p, [0]; color=:grey80, lw=1)
    for (m, se, col) in ((a.hmm, a.hmm_se, C_HMM), (a.tod, a.tod_se, C_TOD))
        plot!(p, a.hour, m .+ se; fillrange=m .- se, color=col, fillalpha=0.2, lw=0)
        plot!(p, a.hour, m; color=col, lw=2)
    end
    s = stats[stats.param .== prm, :]
    plot!(p; title=@sprintf("hour explains %.0f%% of HMM variance · r = %.2f",
                            100median(s.frac_var_hour), median(s.r)), titlefontsize=8)
    p
end
plot!(bottom[1]; ylabel="all rats, centred\n(HMM trial-level SD)")
annotate!(bottom[1], 23.6, 1.04, text("±1 SD", 7, :grey45, :right, :bottom))
let yl = ylims(bottom[3])[2]
    annotate!(bottom[3], 7.5, 0.93yl, text("fed", 7, :grey35, :center, :top))
    annotate!(bottom[3], 18, 0.93yl, text("lights off", 7, :grey35, :center, :top))
end

p = plot(top..., bottom...; layout=(2, 4), size=(1250, 560), left_margin=6Plots.mm,
         bottom_margin=6Plots.mm, dpi=200)
savefig_both(p, joinpath(dir, "param_curves"))
@info "wrote $(joinpath(dir, "param_curves")).{png,svg} (example rat $EXAMPLE)"
