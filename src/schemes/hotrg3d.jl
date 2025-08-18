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
* [Xie et. al. Phys. Rev. B 86 (2012)](@cite xie_coarse-graining_2012)

"""
mutable struct HOTRG_3D <: TNRScheme
    T::TensorMap

    finalize!::Function
    function HOTRG_3D(T::TensorMap{E, S, 2, 4}; finalize = (finalize!)) where {E, S}
        return new(T, finalize)
    end
end

function _step!(scheme::HOTRG_3D, trunc::TensorKit.TruncationScheme)
    # join in z-direction, trace y-indices
    @tensor MMdag1[x1 z z′ x1′] :=
        scheme.T[z z1; y1′ x1c y1 x1] * conj(scheme.T[z′ z1; y1′ x1c y1 x1′])
    @tensor MMdag[x1 x2; x1′ x2′] := scheme.T[z2 z; y2′ x2c y2 x2] *
        conj(scheme.T[z2 z′; y2′ x2c y2 x2′]) * MMdag1[x1 z z′ x1′]
    U, _, _, ε₁ = tsvd(MMdag; trunc)
    _, _, U₂, ε₂ = tsvd(adjoint(MMdag); trunc)
    if ε₁ > ε₂
        U = adjoint(U₂)
    end
    # join in z-direction, trace x-indices
    @tensor MMdag1[y1 z z′ y1′] :=
        scheme.T[z z1; y1c x1′ y1 x1] * conj(scheme.T[z′ z1; y1c x1′ y1′ x1])
    @tensor MMdag[y1 y2; y1′ y2′] := scheme.T[z2 z; y2c x2′ y2 x2] *
        conj(scheme.T[z2 z′; y2c x2′ y2′ x2]) * MMdag1[y1 z z′ y1′]
    V, _, _, ε₁ = tsvd(MMdag; trunc)
    _, _, V₂, ε₂ = tsvd(adjoint(MMdag); trunc)
    if ε₁ > ε₂
        V = adjoint(V₂)
    end
    # apply the truncation
    @tensor scheme.T[-1 -2; -3 -4 -5 -6] := (
        conj(U[x1 x2; -6]) * U[x1′ x2′; -4] *
            conj(V[y1 y2; -5]) * V[y1′ y2′; -3] *
            scheme.T[z -2; y1′ x1′ y1 x1] * scheme.T[-1 z; y2′ x2′ y2 x2]
    )
    return scheme
end

function step!(scheme::HOTRG_3D, trunc::TensorKit.TruncationScheme)
    # z-direction
    _step!(scheme, trunc)
    # x-direction
    scheme.T = permute(scheme.T, ((4, 6), (2, 5, 1, 3)))
    _step!(scheme, trunc)
    # y-direction
    scheme.T = permute(scheme.T, ((4, 6), (2, 5, 1, 3)))
    _step!(scheme, trunc)
    scheme.T = permute(scheme.T, ((4, 6), (2, 5, 1, 3)))
    return scheme
end

function Base.show(io::IO, scheme::HOTRG_3D)
    println(io, "3D HOTRG - Higher Order TRG in 3D")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end
