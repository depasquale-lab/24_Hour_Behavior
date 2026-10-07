#=
Figure 5: when each DDM-HMM state is occupied, and how the states differ.

  A  population occupancy of each state by hour from lights-on (mean ± SEM)
  B  dark-cycle occupancy ratio (hours 12-23 / 24 h)
  C  feeding-time occupancy ratio (hours 6-8 / 24 h)
  D-G  B, |a0 - 0.5|, τ, v by state rank
  H  rat × rank heatmaps of v, τ, |a0 - 0.5| and B (viridis)

States are ranked within animal by posterior-weighted accuracy (1 = most
accurate), the labelling used throughout the paper. Brackets in D-G are paired
t tests between ranks, Holm-corrected over the six pairs, drawn where p < 0.05.

Inputs: state_parameters_long.csv (ExtractStateParameters.jl),
state_occupancy_by_hour.csv (StateOccupancyByHour.jl).
=#

using Pkg
Pkg.activate("notebooks")

using CSV
using DataFrames
using Statistics
using StatsBase
using Distributions
using HypothesisTests
using Printf
using Random
using Plots
using StatsPlots
using PlotUtils: optimize_ticks

gr()
Random.seed!(20260915)   # jitter in panels D-G

# Helvetica so text stays editable in Illustrator. Panels render at their layout
# size (see PANELS); GR uses 100 dpi, 1 px = 0.72 pt.
const FONT_FAMILY = "Helvetica"
default(;
    fontfamily=FONT_FAMILY,
    titlefontfamily=FONT_FAMILY,
    guidefontfamily=FONT_FAMILY,
    tickfontfamily=FONT_FAMILY,
    legendfontfamily=FONT_FAMILY,
    guidefontsize=9,
    tickfontsize=7,
    titlefontsize=12,
    legendfontsize=7,
    grid=false,
    framestyle=:axes,
)

results_dir = joinpath("results", "final_ddmhmms")
params = CSV.read(joinpath(results_dir, "state_parameters_long.csv"), DataFrame)
occ = CSV.read(joinpath(results_dir, "state_occupancy_by_hour.csv"), DataFrame)
params.rat = string.(params.rat)
occ.rat = string.(occ.rat)

const K = 4
const RANK_LABEL = "state rank (1 = most accurate)"
const LAYOUT_PT = (582, 451)                          # Illustrator layout, pt
rank_colors = ["#9E2A2B", "#E29578", "#83C5BE", "#006D77"]   # rank 1 (most accurate) -> 4

transform!(groupby(params, :rat), :acc => (a -> ordinalrank(a; rev=true)) => :acc_rank)
params.bias_mag = abs.(params.a0 .- 0.5)

rats = sort(unique(params.rat))
const NRAT = length(rats)

sem(v) = std(v) / sqrt(length(v))
nanmean(v) = (w = filter(!isnan, v); isempty(w) ? NaN : mean(w))
nansem(v) = (w = filter(!isnan, v); length(w) < 2 ? NaN : sem(w))

# Occupancy: rat × hour × rank

occ_arr = fill(NaN, NRAT, 24, K)
for r in eachrow(occ)
    occ_arr[findfirst(==(r.rat), rats), r.hour + 1, r.acc_rank] = r.occupancy
end

pA = plot(;
    xlabel="hour from lights-on (binned)",
    ylabel="state occupancy",
    xlims=(-0.5, 23.5),
    xticks=0:5:20,
    legend=:topleft,
    foreground_color_legend=:black,
)
let labels = ["1  highest accuracy", "2  high-mid accuracy",
              "3  low-mid accuracy", "4  lowest accuracy"]
    for k in 1:K
        m = [nanmean(occ_arr[:, h, k]) for h in 1:24]
        e = [nansem(occ_arr[:, h, k]) for h in 1:24]
        plot!(pA, 0:23, m; ribbon=e, fillalpha=0.3, color=rank_colors[k],
              linewidth=1.5, label=labels[k])
    end
end

# Occupancy ratio

dark_mask = falses(24);    dark_mask[13:24] .= true   # hours 12-23 from lights-on
feeding_mask = falses(24); feeding_mask[7:9] .= true  # hours 6-8

"Per-rat occupancy in `mask` hours over occupancy across the whole day, rat × rank."
occupancy_ratio(mask) = [nanmean(occ_arr[i, mask, k]) / nanmean(occ_arr[i, :, k])
                    for i in 1:NRAT, k in 1:K]
ratio_dark = occupancy_ratio(dark_mask)
ratio_feed = occupancy_ratio(feeding_mask)

# B and C share one y scale, as in the layout, so their bars compare directly.
ratio_top = 1.15 * maximum(nanmean(E[:, k]) + nansem(E[:, k])
                            for E in (ratio_dark, ratio_feed) for k in 1:K)

function ratio_panel(E; title, ylabel)
    m = [nanmean(E[:, k]) for k in 1:K]
    e = [nansem(E[:, k]) for k in 1:K]
    top = ratio_top
    p = plot(;
        ylabel=ylabel,
        xticks=(1:K, string.(1:K)),
        xlims=(0.4, K + 0.6),
        ylims=(0, top),
        legend=false,
        title=title,
        titlefontsize=9,
    )
    hline!(p, [1.0]; color=:gray45, linewidth=0.6)
    for k in 1:K
        bar!(p, [k], [m[k]]; yerror=[e[k]], fillcolor=rank_colors[k], linecolor=:black,
             linewidth=0.5, bar_width=0.75, markerstrokecolor=:black, markerstrokewidth=0.8)
    end
    return p
end

pB = ratio_panel(ratio_dark; title="dark cycle", ylabel="occupancy ratio")
pC = ratio_panel(ratio_feed; title="feeding time", ylabel="")

# Parameter statistics

"rat × rank matrix of `col`."
wide(col::Symbol) = [only(params[(params.rat .== r) .& (params.acc_rank .== k), col])
                     for r in rats, k in 1:K]

function holm(p::Vector{Float64})
    m = length(p)
    ord = sortperm(p)
    adj = similar(p)
    running = 0.0
    for (i, j) in enumerate(ord)
        running = max(running, min(1.0, (m - i + 1) * p[j]))
        adj[j] = running
    end
    return adj
end

"One-way repeated-measures ANOVA over ranks, subject = rat."
function rm_anova(Y::Matrix{Float64})
    n, k = size(Y)
    g = mean(Y)
    ss_cond = n * sum((mean(Y; dims=1) .- g) .^ 2)
    ss_subj = k * sum((mean(Y; dims=2) .- g) .^ 2)
    ss_err = sum((Y .- g) .^ 2) - ss_subj - ss_cond
    df1, df2 = k - 1, (n - 1) * (k - 1)
    F = (ss_cond / df1) / (ss_err / df2)
    return (; F, df1, df2, p=ccdf(FDist(df1, df2), F))
end

"Paired t tests between every pair of ranks, Holm-corrected."
function posthoc(Y::Matrix{Float64})
    pairs = [(a, b) for a in 1:(K - 1) for b in (a + 1):K]
    tests = [OneSampleTTest(Y[:, b] .- Y[:, a]) for (a, b) in pairs]
    p = pvalue.(tests)
    return DataFrame(a=first.(pairs), b=last.(pairs),
                     diff=[mean(Y[:, b] .- Y[:, a]) for (a, b) in pairs],
                     t=[t.t for t in tests], p=p, p_holm=holm(p))
end

stars(p) = p < 1e-4 ? "****" : p < 1e-3 ? "***" : p < 1e-2 ? "**" : "*"

"""
Box and dots of one parameter by rank. `ymax` clips the axis: points above it
are drawn as open triangles at the top edge, so one outlier cannot squash the
boxes. Nothing is dropped from the tests. `top` overrides the axis top, which
the composer raises by exactly what the significance brackets need.

Returns the plot, the post-hoc table and the bracket spec: the significant
pairs, the top of the drawn data over each rank, and the axis range.
"""
function box_panel(col::Symbol; ylabel, xlabel="", ymin=0.0, ymax=Inf, top=nothing)
    Y = wide(col)
    ph = posthoc(Y)
    sig = ph[ph.p_holm .< 0.05, :]
    rng = MersenneTwister(hash(col))   # jitter fixed per panel, however often it is rebuilt

    lo, hi = min(ymin, minimum(Y)), min(maximum(Y), ymax)
    top = something(top, hi + 0.04 * (hi - lo))

    p = plot(;
        ylabel=ylabel,
        xlabel=xlabel,
        xticks=(1:K, string.(1:K)),
        yticks=filter(t -> t <= top, optimize_ticks(lo, top; k_min=4, k_max=7)[1]),
        xlims=(0.4, K + 0.6),
        ylims=(lo, top),
        legend=false,
    )
    for k in 1:K
        boxplot!(p, fill(k, NRAT), Y[:, k]; fillcolor=rank_colors[k], linecolor=:black,
                 linewidth=0.5, outliers=false, bar_width=0.6, whisker_width=0.6)
        x = k .+ 0.18 .* (rand(rng, NRAT) .- 0.5)
        inside = Y[:, k] .<= ymax
        scatter!(p, x[inside], Y[inside, k]; color=:gray25,
                 markersize=2, markerstrokewidth=0, alpha=0.65)
        # off-scale points sit at the clip value, so the axis can extend above them
        scatter!(p, x[.!inside], fill(min(ymax, top), count(.!inside)); marker=:utriangle, markersize=3,
                 markercolor=:white, markerstrokecolor=:gray25, markerstrokewidth=0.5)
    end
    skyline = vec(maximum(Y; dims=1))
    return p, ph, (; sig, skyline, clipped=skyline .> ymax, ymax, lo, top)
end

"""
Heights for straight significance brackets, packed like the Illustrator layout:
each line sits `gap` above the highest thing under its own span, short brackets
first. Heights are tracked on a half-rank grid (odd index = rank, even =
midpoint). A placed bracket blocks its interior, where its stars sit, by a full
`step`, but its end points only by its own line, so a neighbour that merely
shares an end point sits beside it rather than a tier up.
`skyline[k]` is the top of the drawn data over rank k; any units that grow upward.
"""
function place_brackets(sig::DataFrame, skyline::Vector{Float64}, gap, step)
    sky = repeat(skyline; inner=2)[1:(2K - 1)]
    for m in 2:2:(2K - 2)
        sky[m] = max(sky[m - 1], sky[m + 1])
    end
    y = zeros(nrow(sig))
    for i in sortperm(collect(zip(sig.b .- sig.a, sig.a)))
        lo_, hi_ = 2sig.a[i] - 1, 2sig.b[i] - 1
        y[i] = maximum(sky[lo_:hi_]) + gap
        sky[(lo_ + 1):(hi_ - 1)] .= y[i] + step
        sky[[lo_, hi_]] .= max.(sky[[lo_, hi_]], y[i])
    end
    return y
end

# Blank y labels reserve space; the composer sets the names as live text
# (GR has no τ or subscript zero). τ and v have fixed ranges; one rat at v ≈ 1.9
# is off-scale.
pD, ph_B, br_D = box_panel(:B; ylabel=" ")
pE, ph_bias, br_E = box_panel(:bias_mag; ylabel=" ")
pF, ph_tau, br_F = box_panel(:tau; ylabel=" ", top=1.15)
pG, ph_v, br_G = box_panel(:v; ylabel=" ", ymax=1.35, top=1.5)
BRACKETS = Dict("D" => br_D, "E" => br_E, "F" => br_F, "G" => br_G)

# Heatmaps (colour bar drawn inside each heatmap's axes, same x scale)

"5th-95th percentile colour limits, so one outlier cannot set the scale."
robust_clims(M) = Tuple(quantile(vec(M), (0.05, 0.95)))

const HEAT_CMAP = cgrad(:viridis)
const HEAT_YLIMS = (-4.6, NRAT + 1.7)   # rows 1:NRAT, colour bar and title above, ranks below

rat_order = let g = combine(groupby(params, :rat), :v => mean => :mv)
    sort(g, :mv; rev=true).rat
end
row_of = Dict(r => i for (i, r) in enumerate(rats))

rect(x0, x1, y0, y1) = Shape([x0, x1, x1, x0], [y0, y0, y1, y1])
heat_color(v, cl) = get(HEAT_CMAP, clamp((v - cl[1]) / (cl[2] - cl[1]), 0, 1))

"Filled rectangles drawn as shapes, so placement is exact in data coordinates."
cells!(p, shapes, colors) =
    plot!(p, shapes; seriestype=:shape, fillcolor=permutedims(colors), linewidth=0, linealpha=0)

"Outline the rectangle [x0, x1] × [y0, y1]."
frame!(p, x0, x1, y0, y1) =
    plot!(p, [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0]; color=:black, linewidth=0.5)

heat_params = [:v, :tau, :bias_mag, :B]   # column titles come from the composer
heats = Plots.Plot[]
for col in heat_params
    M = wide(col)[[row_of[r] for r in rat_order], :]
    cl = robust_clims(M)
    hm = plot(; framestyle=:none, xticks=false, yticks=false, legend=false, colorbar=false, yflip=true,
              xlims=(0.5, K + 0.5), ylims=HEAT_YLIMS)

    # matrix
    cells!(hm, [rect(k - 0.5, k + 0.5, i - 0.5, i + 0.5) for i in 1:NRAT for k in 1:K],
           [heat_color(M[i, k], cl) for i in 1:NRAT for k in 1:K])
    frame!(hm, 0.5, K + 0.5, 0.5, NRAT + 0.5)
    for k in 1:K
        annotate!(hm, k, NRAT + 1.2, text(string(k), FONT_FAMILY, 7, :center))
    end

    # colour bar, spanning the matrix width on the same axis
    edges = range(0.5, K + 0.5; length=101)
    vals = range(cl[1], cl[2]; length=100)
    cells!(hm, [rect(edges[m], edges[m + 1], -2.6, -1.6) for m in 1:100],
           [heat_color(v, cl) for v in vals])
    frame!(hm, 0.5, K + 0.5, -2.6, -1.6)
    # two ticks per bar: the matrices are four cells wide, too narrow for more
    cticks = filter(x -> cl[1] <= x <= cl[2], optimize_ticks(cl[1], cl[2]; k_min=2, k_max=4)[1])
    for t in (first(cticks), last(cticks))
        x = 0.5 + K * (t - cl[1]) / (cl[2] - cl[1])
        plot!(hm, [x, x], [-2.6, -2.8]; color=:black, linewidth=0.5)
        annotate!(hm, x, -2.9, text(@sprintf("%g", t), FONT_FAMILY, 6, :center, :bottom))
    end
    push!(heats, hm)
end

# Rat names get their own column on the heatmaps' y scale, so all four heatmaps
# are identical subplots with identical padding.
rat_names = plot(; framestyle=:none, xticks=false, yticks=false, legend=false, yflip=true,
                 xlims=(0, 1), ylims=HEAT_YLIMS)
for (i, r) in enumerate(rat_order)
    annotate!(rat_names, 0.95, i, text(r, FONT_FAMILY, 7, :right))
end
annotate!(rat_names, 0.05, (NRAT + 1) / 2, text("rat", FONT_FAMILY, 9, :center, rotation=90))

# Assemble: each panel rendered alone at its box size in the Illustrator
# layout (Figure_6_24HR_edited.svg), nested into one SVG; letters and shared
# axis labels added as live text

const PANELS = [   # name => (x, y, width, height) in pt
    "A" => (0, 0, 285, 200),
    "B" => (285, 8, 147, 177),
    "C" => (432, 8, 150, 177),
    "D" => (0, 205, 178, 110),
    "E" => (180, 205, 178, 110),
    "F" => (0, 331, 178, 110),   # 16 pt band above F and G takes the brackets
    "G" => (180, 331, 178, 110),
    "H" => (357, 205, 225, 236),
]
PLOTS = Dict("A" => pA, "B" => pB, "C" => pC, "D" => pD, "E" => pE, "F" => pF, "G" => pG,
             "H" => plot(rat_names, heats...; layout=grid(1, 5; widths=fill(0.2, 5))))
const LETTERS = Dict("A" => (2, 13.7), "B" => (290, 13.7), "C" => (436, 13.7),
                     "D" => (2, 216.7), "E" => (181, 216.7), "F" => (2, 342.7),
                     "G" => (180, 342.7), "H" => (357, 216.7))
# Parameter names as SVG markup, so τ and the subscript render in the real font.
const BIAS_SVG = "|a<tspan baseline-shift=\"sub\" font-size=\"70%\">0</tspan>&#160;− 0.5|"
const PARAM_LABELS = [   # (markup, x, y, rotated) in pt
    ("B", 9, 260, true), (BIAS_SVG, 189, 260, true),
    ("τ", 9, 386, true), ("v", 189, 386, true),
    [(lab, 357 + 45 * j + 22.5, 214.0, false)
     for (j, lab) in enumerate(["v", "τ", BIAS_SVG, "B"])]...,
]
const SHARED_LABELS = [   # (text, x centre, baseline y) in pt
    (RANK_LABEL, (285 + 582) / 2, 194.0),
    (RANK_LABEL, 357 / 2, 448.0),
    (RANK_LABEL, (357 + 582) / 2, 448.0),
]

panel_dir = joinpath(results_dir, "figure5_panels")
mkpath(panel_dir)
for p in values(PLOTS)
    plot!(p; margin=0.5Plots.mm)
end

"""
The plot (data) area of a GR panel SVG, in pt: the clip rectangle used by the
most elements, since every series is clipped to it.
"""
function data_rect(svg, w)
    uses = Dict{String,Int}()
    for m in eachmatch(r"url\(#([^)]+)\)", svg)
        uses[m[1]] = get(uses, m[1], 0) + 1
    end
    id = first(sort(collect(uses); by=last, rev=true))[1]
    r = match(Regex("<clipPath id=\"$id\">\\s*<rect x=\"([^\"]+)\" y=\"([^\"]+)\" width=\"([^\"]+)\" height=\"([^\"]+)\""), svg)
    vbw = parse(Float64, match(r"viewBox=\"\S+ \S+ (\S+)", svg)[1])
    return parse.(Float64, r.captures) .* (w / vbw)
end

const PT_PER_MM = 2.835

"Data area (x, y, w, h) of panel `tag` in pt, relative to the panel, from a trial render."
function measure_rect(tag)
    _, _, w, h = Dict(PANELS)[tag]
    p = PLOTS[tag]
    plot!(p; size=round.(Int, (w, h) ./ 0.72), dpi=300)
    savefig(p, joinpath(panel_dir, "_measure.svg"))
    rect = data_rect(read(joinpath(panel_dir, "_measure.svg"), String), w)
    rm(joinpath(panel_dir, "_measure.svg"))
    return rect
end

"""
GR sizes a panel's left margin from its y tick labels, so panels with "0.00"
and "0" ticks do not line up. Pad the narrower margins so every panel in
`tags` has its data area at the same x and of the same width.
"""
function align_left!(tags)
    for tag in tags
        plot!(PLOTS[tag]; left_margin=0.5Plots.mm)
    end
    lefts = Dict(tag => measure_rect(tag)[1] for tag in tags)
    for tag in tags
        plot!(PLOTS[tag]; left_margin=(0.5 + (maximum(values(lefts)) - lefts[tag]) / PT_PER_MM)Plots.mm)
    end
end

# Significance brackets, drawn by the composer over the panel. Each rests
# on the data under it; overflow goes into the band between rows
const BRACKET_GAP, BRACKET_STEP, STAR_PT, STAR_ROOM = 1.5, 5.0, 7, 5.5   # pt

"Bracket heights in pt above the axis bottom, given the data area height `rh`."
function bracket_heights(spec, rh)
    s = rh / (spec.top - spec.lo)
    sky = [(min(v, spec.ymax) - spec.lo) * s + (c ? 2.0 : 0.0)   # 2 pt for the triangle
           for (v, c) in zip(spec.skyline, spec.clipped)]
    return place_brackets(spec.sig, sky, BRACKET_GAP, BRACKET_STEP)
end

"SVG for one panel's brackets, in figure pt."
function bracket_svg(tag)
    spec = BRACKETS[tag]
    isempty(spec.sig) && return String[]
    x0, y0, _, _ = Dict(PANELS)[tag]
    rx, ry, rw, rh = measure_rect(tag)
    X(r) = x0 + rx + (r - 0.4) / (K + 0.2) * rw
    out = String[]
    for (i, h) in enumerate(bracket_heights(spec, rh))
        xa, xb, y = X(spec.sig.a[i]), X(spec.sig.b[i]), y0 + ry + rh - h
        push!(out, """<line x1="$xa" y1="$y" x2="$xb" y2="$y" stroke="#231f20" stroke-width="0.75"/>""",
                   """<text x="$((xa + xb) / 2)" y="$(y - 0.3)" font-family="$FONT_FAMILY" font-size="$STAR_PT" text-anchor="middle">$(stars(spec.sig.p_holm[i]))</text>""")
    end
    return out
end

align_left!(["D", "E", "F", "G"])

"Render `p` at `w` × `h` pt; return the SVG body, its viewBox and ids made unique by `tag`."
function render_panel(p, tag, w, h)
    stem = joinpath(panel_dir, "panel_$tag")
    plot!(p; size=round.(Int, (w, h) ./ 0.72), dpi=300)
    savefig(p, stem * ".svg")
    svg = read(stem * ".svg", String)
    svg = replace(svg, r"<svg([^>]*?) width=\"[^\"]*\" height=\"[^\"]*\"" =>
                       SubstitutionString("<svg\\1 width=\"$(w)pt\" height=\"$(h)pt\""); count=1)
    write(stem * ".svg", svg)   # standalone panel, placeable on its own
    svg = replace(svg, r"id=\"([^\"]+)\"" => SubstitutionString("id=\"$(tag)_\\1\""),
                       r"url\(#([^)]+)\)" => SubstitutionString("url(#$(tag)_\\1)"),
                       r"href=\"#([^\"]+)\"" => SubstitutionString("href=\"#$(tag)_\\1\""))
    vb = match(r"viewBox=\"([^\"]+)\"", svg)[1]
    body = match(r"<svg[^>]*>(.*)</svg>"s, svg)[1]
    return body, vb
end

parts = String[]
for (tag, (x, y, w, h)) in PANELS
    body, vb = render_panel(PLOTS[tag], tag, w, h)
    push!(parts, """<svg x="$x" y="$y" width="$w" height="$h" viewBox="$vb">$body</svg>""")
    haskey(BRACKETS, tag) && append!(parts, bracket_svg(tag))
end
for (tag, (x, y)) in LETTERS
    push!(parts, """<text x="$x" y="$y" font-family="$FONT_FAMILY" font-weight="bold" font-size="12">$tag</text>""")
end
for (label, x, y, rot) in PARAM_LABELS
    pos = rot ? "transform=\"translate($x $y) rotate(-90)\"" : "x=\"$x\" y=\"$y\""
    push!(parts, """<text $pos font-family="$FONT_FAMILY" font-size="9" text-anchor="middle">$label</text>""")
end
for (label, x, y) in SHARED_LABELS
    push!(parts, """<text x="$x" y="$y" font-family="$FONT_FAMILY" font-size="9" text-anchor="middle">$label</text>""")
end

stem = joinpath(results_dir, "figure5_state_ddm")
W, H = LAYOUT_PT
write(stem * ".svg", """<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="$(W)pt" height="$(H)pt" viewBox="0 0 $W $H">
<rect width="$W" height="$H" fill="#ffffff"/>
$(join(parts, "\n"))
</svg>
""")
run(`rsvg-convert -z 4 -b white $(stem * ".svg") -o $(stem * ".png")`)

# Report

println("\nOCCUPANCY RATIO  mean ± SEM over $NRAT rats")
for k in 1:K
    @printf("  rank %d: dark %.2f ± %.2f   feeding %.2f ± %.2f\n", k,
            nanmean(ratio_dark[:, k]), nansem(ratio_dark[:, k]),
            nanmean(ratio_feed[:, k]), nansem(ratio_feed[:, k]))
end

println("\nRM-ANOVA over ranks, then Holm-corrected paired t tests")
for (col, ph) in ((:v, ph_v), (:tau, ph_tau), (:bias_mag, ph_bias), (:B, ph_B))
    a = rm_anova(wide(col))
    eta = a.F * a.df1 / (a.F * a.df1 + a.df2)
    @printf("  %-8s F(%d, %d) = %.2f, p = %.1e, partial η² = %.2f\n",
            col, a.df1, a.df2, a.F, a.p, eta)
    for r in eachrow(sort(ph, :p_holm))
        r.p_holm < 0.05 || continue
        @printf("      rank %d vs %d: Δ = %+.3f, t(%d) = %.2f, p_holm = %.1e %s\n",
                r.a, r.b, r.diff, NRAT - 1, r.t, r.p_holm, stars(r.p_holm))
    end
end
println()
