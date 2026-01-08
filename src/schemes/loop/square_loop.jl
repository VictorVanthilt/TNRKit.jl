"""
$(TYPEDEF)

Loop Optimization for Tensor Network Renormalization

### Constructors
    $(FUNCTIONNAME)(T)
    $(FUNCTIONNAME)(TA, TB)
    $(FUNCTIONNAME)(unitcell_2x2::Matrix{T})

### Running the algorithm
    run!(::LoopTNR, trunc::TensorKit.TruncationScheme, truncentanglement::TensorKit.TruncationScheme, criterion::stopcrit,
              entanglement_criterion::stopcrit, loop_criterion::stopcrit[, finalize_beginning=true, verbosity=1])

    run!(::LoopTNR, trscheme::TensorKit.TruncationScheme, criterion::stopcrit[, finalizer=default_Finalizer, finalize_beginning=true, verbosity=1])

### Fields

$(TYPEDFIELDS)

### References
* [Yang et. al. Phys. Rev. Letters 118 (2017)](@cite yangLoopOptimizationTensor2017)

"""
mutable struct LoopTNR{E, S, TT <: AbstractTensorMap{E, S, 2, 2}} <: LinearLoopScheme{E, S}
    "Central tensor on sublattice A"
    TA::TT

    "Central tensor on sublattice B"
    TB::TT

    function LoopTNR(TA::TT, TB::TT) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
        return new{E, S, TT}(TA, TB)
    end
    function LoopTNR(T::TT) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
        return new{E, S, TT}(T, copy(T))
    end
end

"""
    LoopTNR(
        unitcell_2x2::Matrix{T},
        loop_criterion::stopcrit,
        trunc::TensorKit.TruncationScheme,
        truncentanglement::TensorKit.TruncationScheme
    ) where {T <: AbstractTensorMap{<:Any, <:Any, 2, 2}}

Initialize LoopTNR using a network with 2 x 2 unit cell, 
by first performing one round of loop optimization to reduce
the network to a bipartite one (without normalization). 
"""
function LoopTNR(
        unitcell_2x2::Matrix{T};
        loop_criterion::stopcrit,
        trunc::TensorKit.TruncationScheme,
        truncentanglement::TensorKit.TruncationScheme,
    ) where {T <: AbstractTensorMap{<:Number, <:VectorSpace, 2, 2}}
    ψA = Ψ_A(unitcell_2x2)
    ψB = loop_opt(ψA, loop_criterion, trunc, truncentanglement, 0)
    TA, TB = ΨB_to_TATB(ψB)
    return LoopTNR(TA, TB)
end

# Function to initialize the list of tensors Ψ_A, making it an MPS on a ring
function Ψ_A(unitcell_2x2::Matrix{<:AbstractTensorMap{E, S, 2, 2}}) where {E, S}
    size(unitcell_2x2) == (2, 2) || error("Input unit cell must have 2 x 2 size.")
    ΨA = [
        transpose(unitcell_2x2[1, 1], ((2,), (1, 3, 4)); copy = true),
        transpose(unitcell_2x2[1, 2], ((1,), (3, 4, 2)); copy = true),
        transpose(unitcell_2x2[2, 2], ((3,), (4, 2, 1)); copy = true),
        transpose(unitcell_2x2[2, 1], ((4,), (2, 1, 3)); copy = true),
    ]
    return ΨA
end
function Ψ_A(TA::AbstractTensorMap{E, S, 2, 2}, TB::AbstractTensorMap{E, S, 2, 2}) where {E, S}
    ΨA = [
        transpose(TA, ((2,), (1, 3, 4)); copy = true),
        transpose(TB, ((1,), (3, 4, 2)); copy = true),
        transpose(TA, ((3,), (4, 2, 1)); copy = true),
        transpose(TB, ((4,), (2, 1, 3)); copy = true),
    ]
    return ΨA
end
function Ψ_A(scheme::LoopTNR)
    return Ψ_A(scheme.TA, scheme.TB)
end

# Entanglement filtering function
function entanglement_filtering!(
        scheme::LoopTNR,
        trunc::TensorKit.TruncationScheme,
        entanglement_criterion::stopcrit = default_entanglement_criterion
    )
    scheme.TA, scheme.TB = _entanglement_filtering(
        scheme.TA, scheme.TB, entanglement_criterion, trunc
    )
    return scheme
end

function ΨB_to_TATB(psiB::Vector{T}) where {T <: AbstractTensorMap{<:Any, <:Any, 1, 2}}
    @plansor TA[-1 -2; -3 -4] := psiB[6][-2; 1 2] * psiB[7][2; 3 -4] *
        psiB[2][-3; 3 4] * psiB[3][4; 1 -1]
    @plansor TB[-1 -2; -3 -4] := psiB[1][1; 2 -2] * psiB[4][-4; 2 3] *
        psiB[5][3; 4 -3] * psiB[8][-1; 4 1]
    return TA, TB
end

function loop_opt!(
        scheme::LoopTNR,
        loop_criterion::stopcrit,
        trunc::TensorKit.TruncationScheme,
        truncentanglement::TensorKit.TruncationScheme,
        verbosity::Int
    )
    psiA = Ψ_A(scheme)
    psiB = loop_opt(psiA, loop_criterion, trunc, truncentanglement, verbosity)
    scheme.TA, scheme.TB = ΨB_to_TATB(psiB)
    return scheme
end

function Base.show(io::IO, scheme::LoopTNR)
    println(io, "LoopTNR - Loop Tensor Network Renormalization")
    println(io, "  * TA: $(summary(scheme.TA))")
    println(io, "  * TB: $(summary(scheme.TB))")
    return nothing
end