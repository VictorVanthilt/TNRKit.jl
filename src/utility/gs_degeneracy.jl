"""
    ground_state_degeneracy(T::AbstractTensorMap, unitcell=1)

Compute the Ground State Degeneracy (GSD) from a single network tensor,
using the eigenvalues of the transfer matrix. The GSD is the exponential
of the Shannon entropy of the normalized eigenvalue spectrum.
"""
function ground_state_degeneracy(T::AbstractTensorMap, unitcell::Int = 1)
    indices = Vector{NTuple{4, Int}}(undef, unitcell)
    for i in 1:unitcell
        indices[i] = (i, -i, -(i + unitcell), i + 1)
    end
    indices[end] = (unitcell, -unitcell, -(unitcell + unitcell), 1)

    Ts = fill(T, unitcell)
    Tcontracted = ncon(Ts, indices)

    outinds = ntuple(i -> i, unitcell)
    ininds = ntuple(i -> unitcell + i, unitcell)
    Tcontracted = permute(Tcontracted, (outinds, ininds))

    D, _ = eig_full(Tcontracted)
    D = D / tr(D)
    vals = filter(!iszero, abs.(D.data))
    S = 0.0
    for v in vals
        ev = abs(v)
        if ev > 0
            S -= ev * log(ev)
        end
    end
    return exp(S)
end

"""
    ground_state_degeneracy(TA::AbstractTensorMap, TB::AbstractTensorMap; unitcell=1)

Compute the GSD from a two-site unit cell (TA, TB). Builds an effective
single-site tensor and delegates to the single-tensor method.
"""
function ground_state_degeneracy(TA::AbstractTensorMap, TB::AbstractTensorMap; unitcell::Int = 1)
    norm_const = area_term(TA, TB)
    T1 = TA / abs(norm_const)^(1 / 4)
    T2 = TB / abs(norm_const)^(1 / 4)
    @tensor T_unit[-1 -2; -3 -4] := T1[-1 1; 3 2] * T2[2 6; 4 -3] *
        T2[-2 3; 1 5] * T1[5 4; 6 -4]
    return ground_state_degeneracy(T_unit, unitcell)
end

ground_state_degeneracy(scheme::TNRScheme; unitcell::Int = 1) = ground_state_degeneracy(scheme.T, unitcell)

function ground_state_degeneracy(scheme::BTRG; unitcell::Int = 1)
    @tensor T_unit[-1 -2; -3 -4] := scheme.T[1 2; -3 -4] * scheme.S1[-2; 2] *
        scheme.S2[-1; 1]
    return ground_state_degeneracy(T_unit, unitcell)
end

ground_state_degeneracy(scheme::LoopTNR; unitcell::Int = 2) = ground_state_degeneracy(scheme.TA, scheme.TB; unitcell)

"""
    gu_wen_ratio(T::AbstractTensorMap)

Compute the Gu-Wen ratios (X1, X2) from a single network tensor.
The Gu-Wen ratios are related to the ground state degeneracy and
the scaling dimensions.

# References
* [Zheng-Cheng Gu & Xiao-Gang Wen. PhysRevB.80.155131](@cite gu2009)
* [Satoshi Morita et al. arxiv:2512.03395](@cite morita2025)
"""
function gu_wen_ratio(T::AbstractTensorMap)
    one_norm = norm(@tensor T[1 2; 2 1])
    two_norm_X1 = norm(@tensor T[1 2; 2 3] * T[3 4; 4 1])
    two_norm_X2 = norm(@tensor T[1 2; 3 4] * T[4 3; 2 1])
    X1 = (one_norm^2) / (two_norm_X1)
    X2 = (one_norm^2) / (two_norm_X2)
    return X1, X2
end

"""
    gu_wen_ratio(TA::AbstractTensorMap, TB::AbstractTensorMap)

Compute the Gu-Wen ratios (X1, X2) from a two-site unit cell (TA, TB).
"""
function gu_wen_ratio(TA::AbstractTensorMap, TB::AbstractTensorMap)
    one_norm = norm(
        @tensor opt = true TA[1 2; 3 4] * TB[4 5; 6 1] *
            TB[7 3; 2 8] * TA[8 6; 5 7]
    )
    two_norm_X1 = norm(
        @tensor opt = true TA[1 2; 3 4] * TB[4 5; 6 7] *
            TA[7 8; 9 10] * TB[10 11; 12 1] *
            TB[13 3; 2 14] * TA[14 6; 5 15] * TB[15 9; 8 16] * TA[16 12; 11 13]
    )
    two_norm_X2 = norm(
        @tensor opt = true TA[1 2; 3 4] * TB[4 5; 6 7] *
            TA[7 8; 9 10] * TB[10 11; 12 1] *
            TB[13 9; 2 14] * TA[14 12; 5 15] *
            TB[15 3; 8 16] * TA[16 6; 11 13]
    )
    X1 = (one_norm^2) / (two_norm_X1)
    X2 = (one_norm^2) / (two_norm_X2)
    return X1, X2
end

gu_wen_ratio(scheme::TNRScheme) = gu_wen_ratio(scheme.T)

function gu_wen_ratio(scheme::BTRG)
    @tensor T_unit[-1 -2; -3 -4] := scheme.T[1 2; -3 -4] * scheme.S1[-2; 2] *
        scheme.S2[-1; 1]
    return gu_wen_ratio(T_unit)
end

gu_wen_ratio(scheme::LoopTNR) = gu_wen_ratio(scheme.TA, scheme.TB)
