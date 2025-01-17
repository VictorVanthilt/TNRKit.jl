mutable struct HOTRG <: TRGScheme
    # data
    T::TensorMap

    finalize!::Function
    function HOTRG(T::TensorMap; finalize=finalize!)
        return new(T, finalize)
    end
end

# function step!(scheme::TRG, trunc::TensorKit.TruncationScheme)
#     U, S, V, _ = tsvd(scheme.T, (1, 2), (3, 4); trunc=trunc)
    
#     @plansor begin
#         A[-1 -2; -3] := U[-1 -2; 1]*sqrt(S)[1; -3]
#         B[-1; -2 -3] := sqrt(S)[-1; 1]*V[1; -2 -3]
#     end

#     U, S, V, _ = tsvd(scheme.T, (1, 4), (2, 3); trunc=trunc)

#     # Flip legs to their original domain (to mitigate space mismatch at the end)
#     U = permute(U, (1,), (2, 3))
#     V = permute(V, (1, 2), (3,))

#     @plansor begin
#         C[-1; -2 -3] := U[-1; -2 1]*sqrt(S)[1; -3]
#         D[-1 -2; -3] := sqrt(S)[-1; 1]*V[1 -2; -3]
#     end

#     # @plansor complains here, not sure why
#     @tensor scheme.T[-1 -2; -3 -4] := D[-1 1; 4] * B[-2; 2 1] * C[2; 3 -3] * A[4 3; -4]
#     return scheme
# end

function step!(scheme::HOTRG, trunc::TensorKit.TruncationScheme)
    #HOTRG along horizontal
    @tensor M_temp[-1 -2 -3; -4 -5 -6] := scheme.T[-1 1; -5 -6]*scheme.T[-2 -3; -4 1]

    spaceT = space(scheme.T)
    U = isometry(fuse(spaceT[1], spaceT[1]), spaceT[1] ⊗ spaceT[1])
    Udg = adjoint(U)
    @tensor M[-1 -2; -3 -4] := M_temp[1 2 -2; 3 4 -4]* U[-1; 1 2]*Udg[3 4; -3]

    ML = permute(M, (1,),(2,3,4))
    MLd = adjoint(ML)
    @tensor L[-1; -2] := ML[-1; 1 2 3]*MLd[1 2 3; -2]
    UL, SL, _, _ = tsvd(L; trunc = trunc)

    MR = permute(M, (1,2,4),(3,))
    MRd = adjoint(MR)
    @tensor R[-1,-2] := MRd[-1, 1 2 3]*MR[1 2 3; -2]
    _,SR, UR,_ = tsvd(R; trunc = trunc)

    L1 = @tensor SL[1;1]
    L2 = @tensor SR[1;1]

    if L1 < L2
        PL= UL
        PR = adjoint(UL)
    else 
        PL = adjoint(UR)
        PR = UR
    end

    @tensor scheme.T[-1 -2; -3 -4] := PL[-1; 1]*M[1 -2; 2 -4]*PR[2; -3]


    #HOTRG along vertical
    

    

end