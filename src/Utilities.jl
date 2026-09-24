export aic, bic, train_test_split, synthetictrial, majority_accuracy

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

function synthetictrial(totalFlashes::Int; p_correct::Float64=0.75)
    @assert 0.0 ≤ p_correct ≤ 1.0
    correct_stimulus = rand(Bool) ? 1 : 0

    n_flashes_1 = 0
    for _ in 1:totalFlashes
        # emit the correct stimulus with prob p_correct, else the other
        flash = (rand() < p_correct) ? correct_stimulus : 1 - correct_stimulus
        n_flashes_1 += (flash == 1)
    end
    n_flashes_0 = totalFlashes - n_flashes_1

    response = if n_flashes_0 > n_flashes_1
        0
    elseif n_flashes_1 > n_flashes_0
        1
    else
        (rand(Bool) ? 1 : 0)
    end

    return (response == correct_stimulus)
end

"""
    majority_accuracy(N, p)

Exact optimal accuracy after N bins when the "correct" side flashes with prob p each bin.
Implements A(N; p) = P[K > N/2] + 0.5 * P[K = N/2],  K ~ Binomial(N, p).
"""
function majority_accuracy(N::Integer, p::Real)
    N <= 0 && return 0.5
    B = Binomial(N, p)
    k = fld(N, 2)                          # floor(N/2)
    acc = 1 - cdf(B, k)                    # P[K ≥ k+1] = P[K > N/2]
    if iseven(N)                           # add half of the tie probability when N is even
        acc += 0.5 * pdf(B, k)             # P[K = N/2]
    end
    return acc
end
