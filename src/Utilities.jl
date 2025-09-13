export aic, bic, train_test_split

function aic(k::Int, ll::Real)
    return 2 * (k - ll)
end

function bic(k::Int, n::Int, ll::Real)
    return k * log(n) - 2 * ll
end

function train_test_split(dataset::Vector{Vector}, k::Int)
    n = length(dataset)
    fold_size = Int(n / k)
    
    # Create shuffled indices
    indices = randperm(n)
    
    # Split into k folds
    folds = Vector{Vector{Int}}(undef, k)
    for i in 1:k
        start_idx = (i-1) * fold_size + 1
        end_idx = i == k ? n : i * fold_size  # Handle remainder in last fold
        folds[i] = indices[start_idx:end_idx]
    end
    
    return folds
end