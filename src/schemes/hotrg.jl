mutable struct HOTRG <: TNRScheme
    T::TensorMap

    finalize!::Function
    function HOTRG(T::TensorMap; finalize=finalize!)
        return new(T, finalize)
    end
end

function step!(scheme::HOTRG, trunc::TensorKit.TruncationScheme)
    # Contract along the horizontal direction
    @tensor M[-1 -2 -3; -4 -5 -6] := scheme.T[-1 1; -5 -6] * scheme.T[-2 -3; -4 1]

    # Get unitaries
    U, _, _, εₗ = tsvd(M, ((1, 2), (3, 4, 5, 6)); trunc=trunc)
    _, _, UR, εᵣ = tsvd(M, ((1, 2, 3, 6), (4, 5)); trunc=trunc)

    if εₗ > εᵣ
        U = permute(adjoint(UR), ((2, 1), (3,)))
    end

    # adjoint(U) on the left, U on the right
    @tensor scheme.T[-1 -2; -3 -4] := adjoint(U)[-1; 1 2] * M[1 2 -2; 3 4 -4] * U[4 3; -3]

    return scheme
end

# example convcrit function
hotrg_convcrit(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))

function Base.show(io::IO, scheme::HOTRG)
    println(io, "HOTRG - Higher Order TRG")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end
