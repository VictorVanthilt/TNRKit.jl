"""
$(TYPEDEF)

Tensor Renormalization Group algorithm parameters.

This is a pure algorithm descriptor — it stores truncation and iteration
parameters but no tensor data. Tensors are managed by [`Renormalizer`](@ref).

# Constructors

All parameters are passed as keyword arguments with sensible defaults:

    $(FUNCTIONNAME)(; trunc=truncrank(16), maxiter=20)

# Usage

```julia
alg = TRG()                                          # uses defaults
alg = TRG(; trunc = truncrank(24), maxiter = 25)     # full configuration
renorm = Renormalizer(alg, T)
T_final, norms = run!(renorm)
```

# Fields

$(TYPEDFIELDS)

# References
* [Levin & Nave Phys. Rev. Letters 99(12) (2007)](@cite levin2007)
"""
Base.@kwdef struct TRG <: TNRAlgorithm
    "Truncation strategy for SVD steps"
    trunc::TruncationStrategy = truncrank(16)
    "Maximum number of RG coarse-graining steps"
    maxiter::Int = 20
end

"""
$(TYPEDEF)

Stores the tensor state for one step of a [`TRG`](@ref) renormalization.

$(TYPEDFIELDS)
"""
mutable struct TRGState{TT <: AbstractTensorMap{<:Any, <:Any, 2, 2}}
    "Central tensor"
    T::TT
end

function Renormalizer(alg::TRG, T::TT) where {TT}
    n = norm(@tensor T[1 2; 2 1])
    T_norm = T / n
    state = TRGState{TT}(T_norm)
    return Renormalizer{TRG, TRGState{TT}}(alg, state, [n], 0)
end

"""
    _renorm_step!(r::Renormalizer)

Perform one TRG coarse-graining step followed by trace-based normalization.
Operates on `r` directly to ensure in-place mutation of `r.state.T`.
"""
function _renorm_step!(r::Renormalizer{<:TRG})
    T = r.state.T
    trunc = r.alg.trunc

    # TRG coarse-graining
    A, B = SVD12(T, trunc)
    Tp = transpose(T, ((2, 4), (1, 3)))
    C, D = SVD12(Tp, trunc)
    @plansor T_new[-1 -2; -3 -4] := D[-2; 1 2] * B[-1; 4 1] * C[4 3; -3] * A[3 2; -4]
    r.state.T = T_new

    # Trace-based normalization (same logic as finalize! for TRG)
    n = norm(@tensor r.state.T[1 2; 2 1])
    r.state.T /= n
    push!(r.norms, n)

    return r
end

function get_tensor(r::Renormalizer{<:TRG})
    return r.state.T
end

function Base.show(io::IO, alg::TRG)
    println(io, "TRG - Tensor Renormalization Group")
    println(io, "  * truncation: $(alg.trunc)")
    println(io, "  * maxiter: $(alg.maxiter)")
    return nothing
end
