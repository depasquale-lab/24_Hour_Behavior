@kwdef struct BehaviorTrial{I<:AbstactInt, F<:AbstractFloat}
    ΔFlashes::I # Right Flashes - Left Flashes
    ChooseR::I # 1 -> Chose Right
    Correct::I # 1 -> Correct
    RT::F # # Reaction time
end