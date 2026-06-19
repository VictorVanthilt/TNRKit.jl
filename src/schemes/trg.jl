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

    $(FUNCTIONNAME)(; trunc=truncrank(16), maxiter=20)

# Fields

$(TYPEDFIELDS)

# References
* [Levin & Nave Phys. Rev. Letters 99(12) (2007)](@cite levin2007)
"""
Base.@kwdef struct TRGParams <: TNRParams
    "Truncation strategy for SVD steps"
    trunc::TruncationStrategy = truncrank(16)
    "Maximum number of RG coarse-graining steps"
    maxiter::Int = 20
end

# ==============================================================================
# Renormalizer interface (new iterable API)
# ==============================================================================

function Renormalizer(params::TRGParams, T::TT) where {TT}
    scheme = TRG(T)
    n = finalize!(scheme)
    return Renormalizer{TRGParams, typeof(scheme)}(params, scheme, [n], 0)
end

function Renormalizer(params::TRGParams, scheme::TRG)
    n = finalize!(scheme)
    return Renormalizer{TRGParams, typeof(scheme)}(params, scheme, [n], 0)
end

function get_tensor(r::Renormalizer{<:TRGParams})
    return r.state.T
end

function _renorm_step!(r::Renormalizer{<:TRGParams})
    T = r.state.T
    trunc = r.alg.trunc

    # TRG coarse-graining (Levin & Nave)
    A, B = SVD12(T, trunc)
    Tp = transpose(T, ((2, 4), (1, 3)))
    C, D = SVD12(Tp, trunc)
    @plansor T_new[-1 -2; -3 -4] := D[-2; 1 2] * B[-1; 4 1] * C[4 3; -3] * A[3 2; -4]
    r.state.T = T_new

    # Trace-based normalization
    n = finalize!(r.state)
    push!(r.norms, n)

    return r
end

# ==============================================================================
# Legacy run! — delegates to the new iterable interface
# ==============================================================================

function run!(
        scheme::TRG,
        trscheme::TruncationStrategy,
        criterion::stopcrit;
        verbosity = 1,
    )
    maxit = criterion isa maxiter ? criterion.n :
        criterion isa MultipleCrit ? (c for c in criterion.crits if c isa maxiter) |> first |> (c -> c.n) : 100
    params = TRGParams(; trunc = trscheme, maxiter = maxit)
    renorm = Renormalizer(params, scheme)

    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting simulation\n $(scheme)\n"
        for _ in 1:(renorm.alg.maxiter)
            step!(renorm)
        end
        @infov 1 "Simulation finished after $(renorm.step) RG steps\n"
    end
    return renorm.norms
end

# ==============================================================================
# Base.show
# ==============================================================================

function Base.show(io::IO, scheme::TRG)
    println(io, "TRG - Tensor Renormalization Group")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end

function Base.show(io::IO, params::TRGParams)
    println(io, "TRGParams")
    println(io, "  * truncation: $(params.trunc)")
    println(io, "  * maxiter: $(params.maxiter)")
    return nothing
end
