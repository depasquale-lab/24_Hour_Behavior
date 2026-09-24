export BehaviorTrial
@kwdef struct BehaviorTrial{I<:Int,F<:AbstractFloat}
    ΔFlashes::I # Right Flashes - Left Flashes
    ChooseR::I # 1 -> Chose Right
    Correct::I # 1 -> Correct
    RT::F # # Reaction time
end
