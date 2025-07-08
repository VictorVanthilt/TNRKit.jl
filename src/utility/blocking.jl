function block_tensors(O_list::Matrix)
    m, n = size(O_list)
    ind_list = [[(n + 1) * (i - 1) + j, m * (n + 1) + (m + 1) * (j - 1) + i + 1,
                 m * (n + 1) + (m + 1) * (j - 1) + i,
                 (n + 1) * (i - 1) + j + 1] for i in 1:m for j in 1:n]
    Vv = [space(O_list[end, j])[2] for j in 1:n]
    Uv = isomorphism(fuse(Vv...), prod(Vv))
    Vh = [space(O_list[i, 1])[1] for i in 1:m]
    Uh = isomorphism(fuse(Vh...), prod(Vh))

    ind1 = vcat(-1, [(n + 1) * (i - 1) + 1 for i in 1:m])
    ind2 = vcat(-2, [m * (n + 1) + (m + 1) * j for j in 1:n])
    ind3 = vcat([m * (n + 1) + (m + 1) * (j - 1) + 1 for j in 1:n], -3)
    ind4 = vcat([(n + 1) * i for i in 1:m], -4)
    tensors = vcat(O_list[:], [Uh, Uv, adjoint(Uv), adjoint(Uh)])
    inds = vcat(ind_list, [ind1, ind2, ind3, ind4])
    return permute(ncon(tensors, inds), (1, 2), (3, 4))
end

function coarsegrain(scheme::LoopTNR, trunc::TensorKit.TruncationScheme)
    TA = scheme.TA
    TB = scheme.TB
    dl, ur = SVD12(TA, trunc)
    dr, ul = SVD12(transpose(TB, (2, 4), (1, 3)), trunc)
    @planar T[-1 -2; -3 -4] := ur[-1; 1 4] * dr[1 2; -3] * dl[2 3; -4] * ul[-2; 4 3]
    return T
end

function UD(scheme::LoopTNR, entanglement_criterion::stopcrit, trunc::TensorKit.TruncationScheme)
    TA = scheme.TA
    TB = scheme.TB
    dl, ur = SVD12(TA, trunc)
    dr, ul = SVD12(transpose(TB, (2, 4), (1, 3)), trunc)
    return dl, ur, dr, ul
    
end

function compression(TA::AbstractTensorMap{E,S,2,2}, TB::AbstractTensorMap{E,S,2,2}, entanglement_criterion, trunc::TensorKit.TruncationScheme) where {E, S}
    AU = domain(TA)[1]
    AD = codomain(TA)[2]
    BU = domain(TB)[1]
    BD = codomain(TB)[2]
    ABU = unitary(AU ⊗ BU ← fuse(AU ⊗ BU))
    ABD = unitary(fuse(AD ⊗ BD) ← AD ⊗ BD)
    BAU = unitary(BU ⊗ AU ← fuse(BU ⊗ AU))
    BAD = unitary(fuse(BD ⊗ AD) ← BD ⊗ AD)
    t0 = time()
    @planar opt = true AB[-1 -2; -3 -4] := TA[-1 3; 1 5] * TB[5 4; 2 -4] * ABU[1 2; -3] * ABD[-2; 3 4]
    @planar opt = true BA[-1 -2; -3 -4] := TB[-1 3; 1 5] * TA[5 4; 2 -4] * BAU[1 2; -3] * BAD[-2; 3 4]
    t1 = time()
    println(" Time for initial contraction: ", t1 - t0)
    L_list = [id(fuse(AD ⊗ BD)), id(fuse(BD ⊗ AD))]
    R_list = [id(fuse(AU ⊗ BU)), id(fuse(BU ⊗ AU))]

    PR_list, PL_list = find_projectors([AB, BA], entanglement_criterion, trunc; 
                                       L_list=L_list, R_list=R_list)

    t2 = time()
    println(" Time for finding projectors: ", t2 - t1)
    L, R = SVD12(TA, trunc)

    PU = ABU * PR_list[2]
    PD = PL_list[2] * BAD
    @planar opt = true T[-1 -2; -3 -4] := R[-1; 5 1] * TB[1 2; 3 4] * PU[5 3; -3] * L[4 6; -4] * PD[-2; 2 6]

    println(" Time for final contraction: ", time() - t2)
    return T
end