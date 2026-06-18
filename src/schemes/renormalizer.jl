"""
$(TYPEDEF)

Iterable state machine that performs RG coarse-graining steps on network tensors.

# Fields
$(TYPEDFIELDS)

# Iterator
Each iteration yields `(state, norms)`.
- First yield: normalized initial state before any RG step.
- Subsequent yields: state after each RG step + normalization.
- Stops after `alg.maxiter` RG steps have been performed.
"""
mutable struct Renormalizer{A <: TNRAlgorithm, S}
    "Algorithm configuration (truncation, maxiter, etc.)"
    alg::A
    "Algorithm-specific state holding all network tensors"
    state::S
    "Accumulated normalization factors at each step"
    norms::Vector{Float64}
    "Number of RG steps performed so far"
    step::Int
end

"""
    get_tensor(r::Renormalizer)

Return the tensor(s) stored in the renormalizer's current state.
For a single-tensor scheme like [`TRG`](@ref), returns the central tensor `T`.
"""
function get_tensor end

"""
    _renorm_step!(r::Renormalizer)

Perform one RG step for the specific algorithm.
Dispatches on the algorithm type stored in `r.alg`.
Each algorithm must define its own method.
"""
function _renorm_step! end

function Base.iterate(r::Renormalizer)
    return ((r.state, r.norms), 0)
end

function Base.iterate(r::Renormalizer, state::Int)
    state >= r.alg.maxiter && return nothing
    _renorm_step!(r)
    r.step = state + 1
    return ((r.state, r.norms), state + 1)
end

function Base.show(io::IO, r::Renormalizer)
    println(io, "Renormalizer")
    println(io, "  * algorithm: $(nameof(typeof(r.alg)))")
    println(io, "  * state: $(nameof(typeof(r.state)))")
    println(io, "  * step: $(r.step) / $(r.alg.maxiter)")
    println(io, "  * norms: $(length(r.norms)) entries")
    return nothing
end

"""
    step!(r::Renormalizer)

Perform one RG coarse-graining step. Wraps [`Base.iterate`](@ref).
Throws an error if `maxiter` has already been reached.
"""
function step!(r::Renormalizer)
    r.step >= r.alg.maxiter && error("maxiter ($(r.alg.maxiter)) reached")
    iterate(r, r.step)
    return r
end

"""
    run!(renorm::Renormalizer; verbosity=1)

Run the RG flow to completion. Returns `(final_state, norms)`.
"""
function run!(renorm::Renormalizer; verbosity = 1)
    algname = nameof(typeof(renorm.alg))
    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting $algname simulation\n"
        for (state, norms) in renorm
            @infov 2 "norm: $(norms[end])"
        end
        @infov 1 "Simulation finished after $(renorm.step) RG steps\n"
    end
    return renorm.state, renorm.norms
end
