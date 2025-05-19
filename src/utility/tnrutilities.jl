#Utility functions for QR decomp

function QR_L(L::TensorMap, T::AbstractTensorMap{E,S,1,3}) where {E,S}
    @tensor temp[-1; -2 -3 -4] := L[-1; 1] * T[1; -2 -3 -4]
    _, Rt = leftorth(temp, ((1, 2, 3), (4,)))
    return Rt
end

function QR_R(R::TensorMap, T::AbstractTensorMap{E,S,1,3}) where {E,S}
    @tensor temp[-1; -2 -3 -4] := T[-1; -2 -3 1] * R[1; -4]
    Lt, _ = rightorth(temp, ((1,), (2, 3, 4)))
    return Lt
end

function QR_L(L::TensorMap, T::AbstractTensorMap{E,S,1,2}) where {E,S}
    @tensor temp[-1; -2 -3] := L[-1; 1] * T[1; -2 -3]
    _, Rt = leftorth(temp, ((1, 2), (3,)))
    return Rt
end

function QR_R(R::TensorMap, T::AbstractTensorMap{E,S,1,2}) where {E,S}
    @tensor temp[-1; -2 -3] := T[-1; -2 1] * R[1; -3]
    Lt, _ = rightorth(temp, ((1,), (2, 3)))
    return Lt
end

function QR_L(L::TensorMap, T::AbstractTensorMap{E,S,1,1}) where {E,S}
    @tensor temp[-1; -2] := L[-1; 1] * T[1; -2]
    _, Rt = leftorth(temp, ((1,), (2,)))
    return Rt
end

function QR_R(R::TensorMap, T::AbstractTensorMap{E,S,1,1}) where {E,S}
    @tensor temp[-1; -2] := T[-1; 1] * R[1; -2]
    Lt, _ = rightorth(temp, ((1,), (2,)))
    return Lt
end
#Functions to find the left and right projectors

function find_L(pos::Int, psi::Array, entanglement_criterion::stopcrit)
    L = id(space(psi[pos])[1])
    crit = true
    steps = 0
    error = [Inf]
    n = length(psi)
    while crit
        new_L = copy(L)
        for i in (pos - 1):(pos + n - 2)
            new_L = QR_L(new_L, psi[i % n + 1])
        end
        new_L = new_L / maximum(abs.(new_L.data))

        if space(new_L) == space(L)
            push!(error, abs(norm(new_L - L)))
        end

        L = new_L
        steps += 1
        crit = entanglement_criterion(steps, error)
    end

    return L
end

function find_R(pos::Int, psi::Array, entanglement_criterion::stopcrit)
    n = length(psi)
    if numin(psi[mod(pos - 2, n) + 1]) == 2
        R = id(space(psi[mod(pos - 2, n) + 1])[3]')
    else
        R = id(space(psi[mod(pos - 2, n) + 1])[4]')
    end
    crit = true
    steps = 0
    error = [Inf]
    while crit
        new_R = copy(R)

        for i in (pos - 2):-1:(pos - n - 1)
            new_R = QR_R(new_R, psi[mod(i, n) + 1])
        end
        new_R = new_R / maximum(abs.(new_R.data))

        if space(new_R) == space(R)
            push!(error, abs(norm(new_R - R)))
        end
        R = new_R
        steps += 1
        crit = entanglement_criterion(steps, error)
    end

    return R
end

function P_decomp(R::TensorMap, L::TensorMap, trunc::TensorKit.TruncationScheme)
    @tensor temp[-1; -2] := L[-1; 1] * R[1; -2]
    U, S, V, _ = tsvd(temp, ((1,), (2,)); trunc=trunc)
    re_sq = pseudopow(S, -0.5)

    @tensor PR[-1; -2] := R[-1; 1] * adjoint(V)[1; 2] * re_sq[2; -2]
    @tensor PL[-1; -2] := re_sq[-1; 1] * adjoint(U)[1; 2] * L[2; -2]

    return PR, PL
end

function find_projectors(psi::Array, entanglement_criterion::stopcrit,
                         trunc::TensorKit.TruncationScheme)
    PR_list = []
    PL_list = []
    n = length(psi)
    for i in 1:n
        L = find_L(i, psi, entanglement_criterion)

        R = find_R(i, psi, entanglement_criterion)

        pr, pl = P_decomp(R, L, trunc)

        push!(PR_list, pr)
        push!(PL_list, pl)
    end
    return PR_list, PL_list
end

#cost functions for loop optimisation

function const_C(psiA)
    @tensor tmp[-1 -2; -3 -4] := psiA[1][-2; 1 2 -4] * conj(psiA[1][-1; 1 2 -3])
    for i in 2:4
        @tensor tmp[-1 -2; -3 -4] := tmp[-1 -2; 1 2] * psiA[i][2; 3 4 -4] *
                                     conj(psiA[i][1; 3 4 -3])
    end
    return @tensor tmp[1 2; 1 2]
end

function TNT(pos, psiB)
    @tensor tmp[-1 -2; -3 -4] := psiB[mod(pos, 8) + 1][-2; 1 -4] *
                                 conj(psiB[mod(pos, 8) + 1][-1; 1 -3])
    for i in (pos + 1):(pos + 7)
        @tensor tmp[-1 -2; -3 -4] := tmp[-1 -2; 1 2] * psiB[mod(i, 8) + 1][2; 3 -4] *
                                     conj(psiB[mod(i, 8) + 1][1; 3 -3])
    end
    return @tensor tmp[1 2; 1 2]
end

function WdT(pos, psiA, psiB)
    next_a = mod(ceil(Int, pos / 2), 4) + 1
    next_b = mod(2 * ceil(Int, pos / 2) + 1, 8)

    @tensor tmp[-1 -2; -3 -4] := psiA[next_a][-2; 1 2 -4] * conj(psiB[next_b][-1; 1 3]) *
                                 conj(psiB[next_b + 1][3; 2 -3])
    for i in next_a:(next_a + 2)
        ΨA = psiA[mod(i, 4) + 1]
        ΨB1 = psiB[2 * (mod(i, 4) + 1) - 1]
        ΨB2 = psiB[2 * (mod(i, 4) + 1)]
        @tensor tmp[-1 -2; -3 -4] := tmp[-1 -2; 1 2] * ΨA[2; 3 4 -4] * conj(ΨB1[1; 3 5]) *
                                     conj(ΨB2[5; 4 -3])
    end

    return @tensor tmp[1 2; 1 2]
end

function dWT(pos, psiA, psiB)
    next_a = mod(ceil(Int, pos / 2), 4) + 1
    next_b = mod(2 * ceil(Int, pos / 2) + 1, 8)

    @tensor tmp[-1 -2; -3 -4] := psiB[next_b][-2; 1 2] * psiB[next_b + 1][2; 3 -4] *
                                 conj(psiA[next_a][-1; 1 3 -3])
    for i in next_a:(next_a + 2)
        ΨA = psiA[mod(i, 4) + 1]
        ΨB1 = psiB[2 * (mod(i, 4) + 1) - 1]
        ΨB2 = psiB[2 * (mod(i, 4) + 1)]
        @tensor tmp[-1 -2; -3 -4] := tmp[-1 -2; 1 2] * ΨB1[2; 3 4] * ΨB2[4; 5 -4] *
                                     conj(ΨA[1; 3 5 -3])
    end

    return @tensor tmp[1 2; 1 2]
end

function cost_func(pos, psiA, psiB)
    C = const_C(psiA)
    tNt = TNT(pos, psiB)
    wdt = WdT(pos, psiA, psiB)
    dwt = dWT(pos, psiA, psiB)

    return C + tNt - wdt - dwt
end
