mutable struct ATRG <: TNRScheme
    T::TensorMap

    finalize!::Function
    function ATRG(T::TensorMap{E,S,2,2}; finalize=finalize!) where {E,S}
        return new(T, finalize)
    end
end

function step!(scheme::ATRG, trunc::TensorKit.TruncationScheme)
    U, S, V, _ = tsvd(scheme.T, ((1, 4), (2, 3)); trunc=trunc)

    @tensor A[-1 -2; -3] := U[-1 -3; -2]
    @tensor B[-1; -2 -3] := S[-3; 1] * V[1; -1 -2]
    @tensor C[-1 -2; -3] := U[-1 -3; 1] * S[1; -2]
    @tensor D[-1; -2 -3] := V[-3; -1 -2]

    @tensor M[-1 -2; -3 -4] := B[1; -3 -4] * C[-1 -2; 1]

    U1, S1, V1, _ = tsvd(M, ((1, 4), (2, 3)); trunc=trunc)

    @tensor X[-1 -2; -3] := U1[-1 -3; 1] * sqrt(S1)[1; -2]
    @tensor Y[-1; -2 -3] := sqrt(S1)[-3; 1] * V1[1; -1 -2]

    @tensor temp1[-1; -2 -3 -4] := D[-1; -2 1] * Y[1; -3 -4]
    @tensor temp2[-1 -2 -3; -4] := A[-1 1; -4] * X[-2 -3; 1]

    _, R1 = leftorth(temp1, ((1, 4), (2, 3)))
    R2, _ = rightorth(temp2, ((1, 2), (3, 4)))

    @tensor temp[-1; -2] := R1[-1; 1 2] * R2[2 1; -2]
    U2, S2, V2, _ = tsvd(temp, ((1,), (2,)); trunc=trunc)
    inv_s = pseudopow(S2, -0.5)
    @tensor Proj_1[-1 -2; -3] := R2[-1 -2; 1] * adjoint(V2)[1; 2] * inv_s[2; -3]
    @tensor Proj_2[-1; -2 -3] := inv_s[-1; 1] * adjoint(U2)[1; 2] * R1[2; -2 -3]

    @tensor H[-1; -2 -3] := D[-1; 1 2] * Y[2; 3 -3] * Proj_1[3 1; -2]
    @tensor G[-1 -2; -3] := Proj_2[-1; 1 2] * X[1 -2; 3] * A[2 3; -3]

    @tensor scheme.T[-1 -2; -3 -4] := G[-1 1; -4] * H[-2; -3 1]

    return scheme
end

atrg_convcrit(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))

function Base.show(io::IO, scheme::ATRG)
    println(io, "ATRG - Anisotropic TRG")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end
