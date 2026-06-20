"""
$(TYPEDEF)

Tensor Renormalization Group

# Constructors
    $(FUNCTIONNAME)(T)

# Running the algorithm
    run!(::TRG, trunc::TruncationStrategy, stop::Stopcrit[, finalizer=default_Finalizer, finalize_beginning=true, verbosity=1])

Each step rescales the lattice by a (linear) factor of √2,
and rotate the lattice by 45 degrees in counter clockwise direction.
The elementary modular parameter `τ₀ ↦ (τ₀ - 1) / (τ₀ + 1)`.

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step
    
# Fields

$(TYPEDFIELDS)

# References
* [Levin & Nave Phys. Rev. Letters 99(12) (2007)](@cite levin2007)
"""
mutable struct TRG{E, S, TT <: AbstractTensorMap{E, S, 2, 2}} <: TNRScheme{E, S}
    "central tensor"
    T::TT

    function TRG(T::TT) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
        return new{E, S, TT}(T)
    end
end

"""
$(TYPEDEF)

Parameters for the Tensor Renormalization Group algorithm.

# Constructors

All parameters are passed as keyword arguments with sensible defaults:

    $(FUNCTIONNAME)(; trunc=truncrank(16), stop=maxiter(20))

# Fields

$(TYPEDFIELDS)
"""
Base.@kwdef struct TRGParams <: TNRParams
    "Truncation strategy for SVD steps"
    trunc::TruncationStrategy = truncrank(16)
    "Stopping criterion"
    stop::stopcrit = maxiter(20)
end

function Renormalizer(params::TRGParams, T::TT) where {TT}
    scheme = TRG(T)
    n = finalize!(scheme)
    return Renormalizer(params, scheme, [n])
end

function Renormalizer(params::TRGParams, scheme::TRG)
    n = finalize!(scheme)
    return Renormalizer(params, scheme, [n])
end

function step!(scheme::TRG, trunc::TruncationStrategy)
    A, B = SVD12(scheme.T, trunc)
    Tp = transpose(scheme.T, ((2, 4), (1, 3)))
    C, D = SVD12(Tp, trunc)
    @plansor scheme.T[-1 -2; -3 -4] := D[-2; 1 2] * B[-1; 4 1] * C[4 3; -3] * A[3 2; -4]
    return scheme
end

# TODO: legacy `run!` interface for backward compatibility. To be removed later.
function run!(
        scheme::TRG,
        trscheme::TruncationStrategy,
        criterion::stopcrit;
        verbosity = 1,
    )
    params = TRGParams(; trunc = trscheme, stop = criterion)
    renorm = Renormalizer(params, scheme)
    _, data = run!(renorm; verbosity)
    return data
end

function Base.show(io::IO, scheme::TRG)
    println(io, "TRG - Tensor Renormalization Group")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end

function Base.show(io::IO, params::TRGParams)
    println(io, "TRGParams")
    println(io, "  * truncation: $(params.trunc)")
    println(io, "  * stop: $(params.stop)")
    return nothing
end
