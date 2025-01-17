mutable struct ATRG <: TRGScheme
    T::TensorMap

    finalize!::Function
    function ATRG(T::TensorMap; finalize=finalize!)
        new(T, finalize)
    end
end

function step!(scheme::ATRG, trunc::TensorKit.TruncationScheme)
    U, S, V, _ = tsvd(scheme.T, (1, 4), (2,3); trunc = trunc)

    @tensor A[-1 -2; -3] := U[-1 -3; -2] 
    @tensor B[-1; -2 -3] := S[-3; 1] * V[1; -1 -2]
    @tensor C[-1 -2; -3] := U[-1 -3; 1] * S[1; -2]
    @tensor D[-1; -2 -3] := V[-3; -1 -2]

    @tensor M[-1 -2; -3 -4] := B[1; -3 -4] * C[-1 -2; 1]

    U1, S1, V1, _ = tsvd(M, (1, 4), (2, 3); trunc = trunc)

    @tensor X[-1 -2; -3] := U1[-1 -3; 1] * sqrt(S1)[1; -2]  
    @tensor Y[-1; -2 -3] := sqrt(S1)[-3; 1] * V1[1; -1 -2]

    @tensor temp1[-1; -2 -3 -4] := D[-1; -2 1]* Y[1; -3 -4]
    @tensor temp2[-1 -2 -3; -4] := A[-1 1; -4] *X[-2 -3; 1]

    _, R1 = leftorth(temp1, (1,4), (2,3))
    R2, _ = rightorth(temp2, (1,2), (3,4))

    @tensor temp[-1; -2] := R1[-1; 1 2] * R2[2 1; -2]
    U2, S2, V2, _ = tsvd(temp, (1,), (2,); trunc = trunc)
    inv_s = pseudopow(S2, -0.5)
    @tensor Proj_1[-1 -2; -3] := R2[-1 -2; 1] * adjoint(V2)[1; 2] * inv_s[2; -3]
    @tensor Proj_2[-1; -2 -3] := inv_s[-1; 1] * adjoint(U2)[1; 2] * R1[2; -2 -3]

    @tensor H[-1; -2 -3] := D[-1; 1 2]* Y[2; 3 -3] *Proj_1[3 1; -2]
    @tensor G[-1 -2; -3] := Proj_2[-1; 1 2]* X[1 -2; 3] * A[2 3; -3]

    @tensor scheme.T[-1 -2; -3 -4] := G[-1 1; -4] * H[-2; -3 1]

    return scheme
end

function finalize!(scheme::HOTRG_robust)
    n = norm(@tensor scheme.T[1 2; 1 2])
    scheme.T /= n

    # turn the tensor by 90 degrees
    scheme.T = permute(scheme.T, ((2, 3), (4, 1)))

    return n
end

atrg_convcrit(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))