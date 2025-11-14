using Pkg
Pkg.activate("notebooks")

using Random
using DriftDiffusionModels
using HiddenMarkovModels
using Dates
using CSV
using DataFrames
using Statistics

# set random seed 
Random.seed!(67)  # this seed is bussin fr fr on god

# set data path (assumes you are in the project root)
data_file = joinpath("data", "processed_rat_data.csv.gz")

# Read in and structure data
rat_df = CSV.read(data_file, DataFrame)
rat_df = rat_df[rat_df.daily .== "24 hr", :] # only keep 24 hour data

# preprocess the data to have numerics
replace!(rat_df[!, :choose_right], 0 => -1)

mapping = Dict("right" => 1, "left" => -1)
DataFrames.transform!(rat_df, :correct_side => ByRow(cs -> get(mapping, cs, missing)) => :correct_side_numeric)

names = unique(rat_df[!, "name"])

rat = names[9]

rat_of_interest = rat_df[rat_df.name .== rat, :]
dates = [Date(split(dt)[1]) for dt in rat_of_interest.trial_datetime]
	
# Get unique dates in chronological order
unique_dates = sort(unique(dates))
	
# Create a vector of vectors, where each inner vector contains DDMResults for one day
results_by_date = Vector{Vector{DDMResult}}()
	
for date in unique_dates
	# Get indices for this date
	day_indices = findall(dates .== date)

	# Skip days with no valid data
	if isempty(day_indices)
		continue
	end

	# Extract RTs and outcomes for this date
	day_rts = rat_of_interest.rt[day_indices]
	day_outcomes = rat_of_interest.choose_right[day_indices]
	day_stim_side = rat_of_interest.correct_side_numeric[day_indices]
	    
	# Create DDMResult objects for this day
	day_results = [DDMResult(rt, choice, stim) for (rt, choice, stim) in zip(day_rts, day_outcomes, day_stim_side)]
	
	# Add to our vector of vectors
	push!(results_by_date, day_results)
end 

# K-Fold Cross Validation with normalized likelihood
function kfold_cv(results_by_date::Vector{Vector{DDMResult}}, 
                  hmm_init::PriorHMM, 
                  k_folds::Int=5, 
                  max_iter::Int=75)::Tuple{Float64, Float64, Vector{Float64}}
    
    n_sessions::Int = length(results_by_date)
    fold_size::Int = n_sessions / k_folds
    
    cv_scores = Float64[]
    
    for fold in 1:k_folds
        # Define test indices
        test_start::Int = (fold - 1) * fold_size + 1
        test_end::Int = fold == k_folds ? n_sessions : fold * fold_size
        test_idx = test_start:test_end
        
        # Split data
        train_sessions = [results_by_date[i] for i in 1:n_sessions if i ∉ test_idx]
        test_sessions = [results_by_date[i] for i in test_idx]
        
        train_data::Vector{DDMResult} = reduce(vcat, train_sessions)
        test_data::Vector{DDMResult} = reduce(vcat, test_sessions)
        
        train_seq_ends::Vector{Int} = cumsum([length(s) for s in train_sessions])
        test_seq_ends::Vector{Int} = cumsum([length(s) for s in test_sessions])
        
        # Train on fold
        hmm_fold, train_lls = baum_welch(hmm_init, train_data; 
                                         seq_ends=train_seq_ends, max_iterations=max_iter)
        
        # Test: compute log-likelihood on held-out data
        γ, test_ll = forward_backward(hmm_fold, test_data; seq_ends=test_seq_ends)
        
        # Normalize by number of test trials for fair comparison
        n_test_trials::Int = length(test_data)
        normalized_ll::Float64 = sum(test_ll) / n_test_trials
        
        push!(cv_scores, normalized_ll)
    end
    
    mean_ll::Float64 = mean(cv_scores)
    std_ll::Float64 = std(cv_scores)
    
    return mean_ll, std_ll, cv_scores
end