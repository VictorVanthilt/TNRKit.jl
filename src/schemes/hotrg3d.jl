"""
$(TYPEDEF)

3D Higher-Order Tensor Renormalization Group

### Constructors
    $(FUNCTIONNAME)(T [, finalize=finalize!])

### Running the algorithm
    run!(::HOTRG_3D, trunc::TensorKit.TruncationSheme, stop::Stopcrit[, finalize_beginning=true, verbosity=1])

Each step rescales the lattice by a (linear) factor of 2

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

### Fields

$(TYPEDFIELDS)

### References
* [Xie et. al. Phys. Rev. B 86 (2012)](@cite xieCoarsegrainingRenormalizationHigherorder2012)

"""
mutable struct HOTRG_3D <: TNRScheme
    T::TensorMap

    finalize!::Function
    function HOTRG_3D(T::TensorMap{E, S, 2, 4}; finalize = (finalize!)) where {E, S}
        return new(T, finalize)
    end
end

function _step_hotrg3d(
        A1::TensorMap{E, S, 2, 4}, A2::TensorMap{E, S, 2, 4},
        trunc::TensorKit.TruncationScheme
    ) where {E, S}
    # join in z-direction, keep x-indices open (A1 below A2)
    @tensoropt MMdag2[x2 z z′ x2′] :=
        A2[z z2; Y2 X2 y2 x2] * conj(A2[z′ z2; Y2 X2 y2 x2′])
    @tensoropt MMdag[x1 x2; x1′ x2′] := MMdag2[x2 z z′ x2′] *
        A1[z1 z; Y1 X1 y1 x1] * conj(A1[z1 z′; Y1 X1 y1 x1′])
    TensorKit.normalize!(MMdag, Inf)
    U, s₁, _, ε₁ = tsvd(MMdag; trunc)
    _, s₂, U₂, ε₂ = tsvd(adjoint(MMdag); trunc)
    @debug "SVD of MM† (for x-truncation)" singular_values = s₁ trunc_err = ε₁
    @debug "SVD of M†M (for x-truncation)" singular_values = s₂ trunc_err = ε₂
    if ε₁ > ε₂
        U = adjoint(U₂)
    end
    # join in z-direction, keep y-indices open
    @tensoropt MMdag2[y2 z z′ y2′] :=
        A2[z z2; Y2 X2 y2 x2] * conj(A2[z′ z2; Y2 X2 y2′ x2])
    @tensoropt MMdag[y1 y2; y1′ y2′] := MMdag2[y2 z z′ y2′] *
        A1[z1 z; Y1 X1 y1 x1] * conj(A1[z1 z′; Y1 X1 y1′ x1])
    TensorKit.normalize!(MMdag, Inf)
    V, s₁, _, ε₁ = tsvd(MMdag; trunc)
    _, s₂, V₂, ε₂ = tsvd(adjoint(MMdag); trunc)
    @debug "SVD of MM† (for y-truncation)" singular_values = s₁ trunc_err = ε₁
    @debug "SVD of M†M (for y-truncation)" singular_values = s₂ trunc_err = ε₂
    if ε₁ > ε₂
        V = adjoint(V₂)
    end
    # apply the truncation
    @tensoropt T[-1 -2; -3 -4 -5 -6] :=
        conj(U[x1 x2; -6]) * U[x1′ x2′; -4] *
        conj(V[y1 y2; -5]) * V[y1′ y2′; -3] *
        A1[-1 z; y1′ x1′ y1 x1] * A2[z -2; y2′ x2′ y2 x2]
    return T
end

function _step!(scheme::HOTRG_3D, trunc::TensorKit.TruncationScheme)
    scheme.T = _step_hotrg3d(scheme.T, scheme.T, trunc)
    return scheme
end

function step!(scheme::HOTRG_3D, trunc::TensorKit.TruncationScheme)
    _step!(scheme, trunc)
    scheme.T = permute(scheme.T, ((6, 4), (2, 3, 1, 5)))
    _step!(scheme, trunc)
    scheme.T = permute(scheme.T, ((6, 4), (2, 3, 1, 5)))
    _step!(scheme, trunc)
    scheme.T = permute(scheme.T, ((6, 4), (2, 3, 1, 5)))
    return scheme
end

function Base.show(io::IO, scheme::HOTRG_3D)
    println(io, "3D HOTRG - Higher Order TRG in 3D")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end
