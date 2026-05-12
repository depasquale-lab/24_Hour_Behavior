### A Pluto.jl notebook ###
# v0.20.22

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ face0002-0000-4000-8000-000000000002
begin
    import Pkg
    Pkg.activate(@__DIR__)

    using CSV
    using DataFrames
    using Dates
    using PlutoUI
    using Plots
    using Statistics
    nothing
end

# ╔═╡ face0001-0000-4000-8000-000000000001
md"""
# Miscellaneous figures

Standalone figures derived from the raw trial-level data: trial-time
histogram (time-from-light-on) and per-rat daily accuracy curves.

| § | Section |
|---|---------|
| 1 | Setup & data loading |
| 2 | Trial-time histogram |
| 3 | Daily accuracy curves |
| 4 | Save figures |
"""

# ╔═╡ face0003-0000-4000-8000-000000000003
md"""
## §1 Setup & data loading
"""

# ╔═╡ face0004-0000-4000-8000-000000000004
begin
    rat_df = CSV.read(joinpath(@__DIR__, "..", "data", "processed_rat_data.csv.gz"), DataFrame)
    rat_df = rat_df[rat_df.daily .== "24 hr", :]
    rat_df
end

# ╔═╡ face0005-0000-4000-8000-000000000005
md"""
## §2 Trial-time histogram

Distribution of trial times relative to light-on (07:30 offset). Bins span
the 24-hour cycle.
"""

# ╔═╡ face0006-0000-4000-8000-000000000006
begin
    trial_datetimes = DateTime.(rat_df.trial_datetime, dateformat"yyyy-mm-dd HH:MM:SS")
    times = Time.(trial_datetimes)

    # Wrap to "time from light on" using a 7h30 offset.
    offset  = Hour(7) + Minute(30)
    refdate = Date(2000, 1, 1)
    wrapped_times = Time.(DateTime.(refdate, times) .- offset)

    time_hours = hour.(wrapped_times) .+
                 minute.(wrapped_times) ./ 60 .+
                 second.(wrapped_times) ./ 3600
    nothing
end

# ╔═╡ face0007-0000-4000-8000-000000000007
trial_time_plot = histogram(
    time_hours;
    fontfamily = "helvetica",
    bins       = 48,
    xlabel     = "Time from light on (H)",
    ylabel     = "Count",
    xlims      = (0, 24),
    legend     = false,
)

# ╔═╡ face0008-0000-4000-8000-000000000008
md"""
## §3 Daily accuracy curves

Per-rat daily accuracy traces (light grey) with the across-rat mean (bold).
Only rats with > 80 daily sessions are kept.
"""

# ╔═╡ face0009-0000-4000-8000-000000000009
begin
    df = deepcopy(rat_df)
    df.trial_datetime = DateTime.(df.trial_datetime, dateformat"y-m-d H:M:S")
    df.session_date   = Date.(df.trial_datetime)
    df.correct        = Float64.(df.correct)
    df.rt             = Float64.(df.rt)

    daily = combine(groupby(df, [:name, :session_date]),
        nrow    => :trials,
        :correct => mean => :acc,
        :rt      => mean => :rt_mean,
    )

    days_per_rat = combine(groupby(daily, :name), nrow => :ndays)
    valid_rats   = days_per_rat.name[days_per_rat.ndays .> 80]
    daily        = daily[in.(daily.name, Ref(valid_rats)), :]

    daily.day = similar(daily.session_date, Int)
    for sub in groupby(daily, :name)
        idxs = parentindices(sub)[1]
        daily.day[idxs] = collect(1:length(idxs))
    end

    mean_by_day = sort!(combine(groupby(daily, :day),
        :trials  => mean => :trials_mean,
        :acc     => mean => :acc_mean,
        :rt_mean => mean => :rt_mean_mean,
    ), :day)
    (size(daily), size(mean_by_day))
end

# ╔═╡ face000a-0000-4000-8000-00000000000a
function plot_daily_metric(metric::Symbol, mean_metric::Symbol;
                           ylabel::AbstractString = "",
                           title::AbstractString  = "",
                           clamp_acc::Bool        = false)
    p = plot(legend = false, xlabel = "Day",
             fontfamily = "helvetica",
             ylabel = ylabel, title = title)
    for sub in groupby(daily, :name)
        plot!(p, sub.day, sub[!, metric];
              color = :lightgrey, lw = 1, alpha = 0.9)
    end
    clamp_acc && ylims!(p, 0.5, 1.0)
    plot!(p, mean_by_day.day, mean_by_day[!, mean_metric]; lw = 3)
    return p
end

# ╔═╡ face000b-0000-4000-8000-00000000000b
daily_acc_plot = plot_daily_metric(:acc, :acc_mean;
                                   ylabel    = "Accuracy",
                                   clamp_acc = true)

# ╔═╡ face000c-0000-4000-8000-00000000000c
daily_rt_plot = plot_daily_metric(:rt_mean, :rt_mean_mean;
                                  ylabel = "RT (mean, s)",
                                  title  = "Daily RT")

# ╔═╡ face000d-0000-4000-8000-00000000000d
daily_trials_plot = plot_daily_metric(:trials, :trials_mean;
                                      ylabel = "Trials/day",
                                      title  = "Daily trials")

# ╔═╡ face000e-0000-4000-8000-00000000000e
md"""
## §4 Save figures
"""

# ╔═╡ face000f-0000-4000-8000-00000000000f
md"""
Save all figures to `../results/`: $(@bind save_figs CheckBox(default = false))
"""

# ╔═╡ face0010-0000-4000-8000-000000000010
begin
    if save_figs
        out_dir = joinpath(@__DIR__, "..", "results")
        mkpath(out_dir)
        figs = [
            (trial_time_plot,   "rat_trial_time_histogram.svg"),
            (daily_acc_plot,    "rat_daily_acc.svg"),
            (daily_rt_plot,     "rat_daily_rt.svg"),
            (daily_trials_plot, "rat_daily_trials.svg"),
        ]
        for (fig, name) in figs
            savefig(fig, joinpath(out_dir, name))
        end
        md"Saved $(length(figs)) figures to `$out_dir`."
    else
        md"_Tick the box above to save all figures._"
    end
end

# ╔═╡ Cell order:
# ╟─face0001-0000-4000-8000-000000000001
# ╠═face0002-0000-4000-8000-000000000002
# ╟─face0003-0000-4000-8000-000000000003
# ╠═face0004-0000-4000-8000-000000000004
# ╟─face0005-0000-4000-8000-000000000005
# ╠═face0006-0000-4000-8000-000000000006
# ╠═face0007-0000-4000-8000-000000000007
# ╟─face0008-0000-4000-8000-000000000008
# ╠═face0009-0000-4000-8000-000000000009
# ╠═face000a-0000-4000-8000-00000000000a
# ╠═face000b-0000-4000-8000-00000000000b
# ╠═face000c-0000-4000-8000-00000000000c
# ╠═face000d-0000-4000-8000-00000000000d
# ╟─face000e-0000-4000-8000-00000000000e
# ╟─face000f-0000-4000-8000-00000000000f
# ╠═face0010-0000-4000-8000-000000000010
