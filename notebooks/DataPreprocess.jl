using CSV
using DataFrames
using CodecZlib

function preprocess_data(df::DataFrame; outdir::String=joinpath(@__DIR__, "..", "data"))
    println("Preprocessing data...")

    # 1) Count flashes from binary strings like "101" -> 2, "010" -> 1
    count_ones = x -> x === missing ? missing : count(==('1'), x)
    df[!, :flashes_left] = map(count_ones, df[!, :flashes_left])
    df[!, :flashes_right] = map(count_ones, df[!, :flashes_right])

    # 2) Drop omissions (if any)
    filter!(row -> lowercase(strip(String(row.choice))) != "omission", df)

    # 3) delta_flashes = right - left
    df[!, :delta_flashes] = df[!, :flashes_right] .- df[!, :flashes_left]

    # 4) choose_right: 1 if choice == "right", else 0
    df[!, :choose_right] = map(df[!, :choice]) do x
        x === missing ? missing : (lowercase(strip(x)) == "right" ? 1 : 0)
    end

    # 5) correct: 1 if outcome == "correct", else 0
    df[!, :correct] = map(df[!, :outcome]) do x
        x === missing ? missing : (lowercase(strip(x)) == "correct" ? 1 : 0)
    end

    println("Data preprocessing complete.")

    # 6) Save as gzipped CSV for GitHub
    mkpath(outdir)
    csv_gz_path = joinpath(outdir, "processed_rat_data.csv.gz")
    open(csv_gz_path, "w") do io
        gz = GzipCompressorStream(io)
        CSV.write(gz, df)
        close(gz)
    end

    println("Saved compressed CSV -> ", csv_gz_path)
    return df
end

# Usage
data_path = joinpath(@__DIR__, "..", "data", "rat_data.csv")
raw_data = CSV.read(
    data_path,
    DataFrame;
    types=Dict(:flashes_left => String, :flashes_right => String),
    stringtype=String,
)

processed = preprocess_data(copy(raw_data))
