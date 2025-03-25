mutable struct HOTRG <: TNRScheme
    T::TensorMap

    finalize!::Function
    function HOTRG(T::TensorMap{E,S,2,2}; finalize=finalize!) where {E,S}
        return new(T, finalize)
    end
end

function step!(scheme::HOTRG, trunc::TensorKit.TruncationScheme)
    @tensor MMdag[-1 -2; -3 -4] := scheme.T[-1 5; 2 1] * scheme.T[-2 3; 4 5] *
                                   adjoint(scheme.T)[4 6; -4 3] *
                                   adjoint(scheme.T)[2 1; -3 6]

    # Get unitaries
    U, _, _, εₗ = tsvd(MMdag; trunc=trunc)
    _, _, Uᵣ, εᵣ = tsvd(adjoint(MMdag); trunc=trunc)

    if εₗ > εᵣ
        U = adjoint(Uᵣ)
    end

    # adjoint(U) on the left, U on the right
    @tensor scheme.T[-1 -2; -3 -4] := adjoint(U)[-1; 1 2] * scheme.T[1 5; 4 -4] *
                                      scheme.T[2 -2; 3 5] * U[4 3; -3]
    return scheme
end

# example convcrit function
hotrg_convcrit(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))

function Base.show(io::IO, scheme::HOTRG)
    println(io, "HOTRG - Higher Order TRG")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end
