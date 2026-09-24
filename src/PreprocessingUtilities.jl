export tag_sessions!, summarize_sessions, to_glmhmm_sequences

# 1) Tag sessions (midnight→midnight), order trials, add within-session indices
function tag_sessions!(
    df::DataFrame;
    subject::Symbol=:name,
    tcol::Symbol=:trial_datetime,
    trialcol::Symbol=:trial,
    parse_format::DateFormat=dateformat"yyyy-mm-dd HH:MM:SS",
)
    df = copy(df)  # keep original intact

    # Parse time if needed
    if !(eltype(df[!, tcol]) <: DateTime)
        df[!, tcol] = DateTime.(df[!, tcol], parse_format)
    end

    # Sort in a stable, sensible way
    sort!(df, [subject, tcol, trialcol])

    # Midnight→midnight sessions: use the calendar Date
    df[!, :session_date] = Date.(df[!, tcol])

    # Group by subject & session day
    g = groupby(df, [subject, :session_date])

    # Assign a session_id (1,2,3,...) in group order
    df[!, :session_id] = vcat((fill(i, nrow(s)) for (i, s) in enumerate(g))...)

    # Index trials within each session (1..n)
    df[!, :t_in_session] = vcat((collect(1:nrow(s)) for s in g)...)

    return df, g
end

# 2) Build a reusable session summary table for plotting/QA
function summarize_sessions(
    g::GroupedDataFrame;
    tcol::Symbol=:trial_datetime,
    rtcol::Symbol=:rt,
    outcol::Symbol=:outcome,                 # still supported if you use it elsewhere
    choose_right_col::Union{Symbol,Nothing}=:choose_right,  # add prop_right if present
)
    # Core aggregations
    aggs = Any[
        nrow => :n_trials,
        tcol => first => :start_time,
        tcol => last => :end_time,
        tcol => (x -> last(x) - first(x)) => :duration,
        rtcol => mean => :mean_rt,
    ]
    # Optionally include proportion of right choices
    if choose_right_col !== nothing && hasproperty(parent(g), choose_right_col)
        push!(aggs, choose_right_col => (v -> mean((v .!= 0))) => :prop_right)
    end

    sessions = combine(g, aggs...)

    # Keep a stable session_id that matches group order
    sessions[!, :session_id] = 1:nrow(sessions)

    # Nice key for labeling plots, e.g. "Daenerys • 2021-06-22"
    if hasproperty(sessions, :name) && hasproperty(sessions, :session_date)
        sessions[!, :session_label] = string.(sessions.name, " • ", sessions.session_date)
    end
    return sessions
end

function to_glmhmm_sequences(
    df::DataFrame;
    subject::Symbol=:name,
    session_id::Symbol=:session_id,
    features::Vector{Symbol}=[:delta_flashes],  # you can pass more later
    target::Symbol=:choose_right,               # numeric {0,1} (or nonzero→1)
)
    g = groupby(df, [subject, session_id])
    seqs = Vector{Vector{GLMObs}}(undef, length(g))

    for (i, s) in enumerate(g)
        X = Matrix{Float64}(select(s, features))     # N×P
        y = Int.(s[!, target] .!= 0)                 # N

        seqs[i] = [GLMObs(vec(X[j, :]), y[j]) for j in 1:size(X, 1)]
    end
    return seqs
end
