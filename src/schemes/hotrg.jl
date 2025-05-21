"""
$(TYPEDEF)

Higher-Order Tensor Renormalization Group

    HOTRG(T [, finalize=finalize!])

### Fields

$(TYPEDFIELDS)

### References
* [Xie et. al. Phys. Rev. B 86 (2012)](@cite xie_coarse-graining_2012)

"""
mutable struct HOTRG <: TNRScheme
    T::TensorMap

    finalize!::Function
    function HOTRG(T::TensorMap{E,S,2,2}; finalize=(finalize!)) where {E,S}
        return new(T, finalize)
    end
end

function step!(scheme::HOTRG, trunc::TensorKit.TruncationScheme)
    # join vertically
    @tensor MMdag[-1 -2; -3 -4] := scheme.T[-1 5; 1 2] * scheme.T[-2 3; 5 4] *
                                   conj(scheme.T[-3 6; 1 2]) * conj(scheme.T[-4 3; 6 4])

    # get unitaries
    U, _, _, εₗ = tsvd(MMdag; trunc=trunc)
    _, _, Uᵣ, εᵣ = tsvd(adjoint(MMdag); trunc=trunc)

    if εₗ > εᵣ
        U = adjoint(Uᵣ)
    end

    # adjoint(U) on the left, U on the right
    @tensor scheme.T[-1 -2; -3 -4] := scheme.T[1 5; -3 3] * conj(U[1 2; -1]) * U[3 4; -4] *
                                      scheme.T[2 -2; 5 4]

    # join horizontally
    @tensor MMdag[-1 -2; -3 -4] := scheme.T[1 -1; 2 5] * scheme.T[5 -2; 4 3] *
                                   conj(scheme.T[1 -3; 2 6]) * conj(scheme.T[6 -4; 4 3])

    # get unitaries
    U, _, _, εₗ = tsvd(MMdag; trunc=trunc)
    _, _, Uᵣ, εᵣ = tsvd(adjoint(MMdag); trunc=trunc)

    if εₗ > εᵣ
        U = adjoint(Uᵣ)
    end

    # adjoint(U) on the bottom, U on top
    @tensor scheme.T[-1 -2; -3 -4] := scheme.T[-1 1; 3 5] * scheme.T[5 2; 4 -4] *
                                      conj(U[1 2; -2]) *
                                      U[3 4; -3]
    return scheme
end

function Base.show(io::IO, scheme::HOTRG)
    println(io, "HOTRG - Higher Order TRG")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end
