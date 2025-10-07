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

function _get_hotrg3d_xproj(
        A1::TensorMap{E, S, 2, 4}, A2::TensorMap{E, S, 2, 4},
        trunc::TensorKit.TruncationScheme
    ) where {E, S}
    # join in z-direction, keep x-indices open (A1 below A2)
    # left unitary
    A2′ = twistdual(A2, [2, 3, 4, 5])
    A1′ = twistdual(A1, [1, 3, 4, 5])
    @tensoropt MM[x2 z z′ x2′] :=
        A2[z z2; Y2 X2 y2 x2] * conj(A2′[z′ z2; Y2 X2 y2 x2′])
    @tensoropt MM[x1 x2; x1′ x2′] := MM[x2 z z′ x2′] *
        A1[z1 z; Y1 X1 y1 x1] * conj(A1′[z1 z′; Y1 X1 y1 x1′])
    TensorKit.normalize!(MM, Inf)
    U, s, _, ε = tsvd!(MM; trunc)
    @debug "Ux left truncation" singular_values = s trunc_err = ε
    # right unitary
    A2′ = twistdual(A2, [2, 3, 5, 6])
    A1′ = twistdual(A1, [1, 3, 5, 6])
    @tensoropt MM[x2 z z′ x2′] :=
        conj(A2[z z2; Y2 x2 y2 X2]) * A2′[z′ z2; Y2 x2′ y2 X2]
    @tensoropt MM[x1 x2; x1′ x2′] := MM[x2 z z′ x2′] *
        conj(A1[z1 z; Y1 x1 y1 X1]) * A1′[z1 z′; Y1 x1′ y1 X1]
    TensorKit.normalize!(MM, Inf)
    _, s′, U′, ε′ = tsvd!(MM; trunc)
    @debug "Ux right truncation" singular_values = s′ trunc_err = ε′
    if ε > ε′
        U, s, ε = adjoint(U′), s′, ε′
    end
    return U, s, ε
end

function _step_hotrg3d(
        A1::TensorMap{E, S, 2, 4}, A2::TensorMap{E, S, 2, 4},
        trunc::TensorKit.TruncationScheme
    ) where {E, S}
    Ux, = _get_hotrg3d_xproj(A1, A2, trunc)
    # switch x/y directions
    perm = ((1, 2), (4, 3, 6, 5))
    Uy, = _get_hotrg3d_xproj(permute(A1, perm), permute(A2, perm), trunc)
    # apply the truncation
    @tensoropt T[-1 -2; -3 -4 -5 -6] :=
        conj(Ux[x1 x2; -6]) * Ux[x1′ x2′; -4] *
        conj(Uy[y1 y2; -5]) * Uy[y1′ y2′; -3] *
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
