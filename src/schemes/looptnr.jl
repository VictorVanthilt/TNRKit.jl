#TODO: Add documentation
mutable struct LoopTNR <: TNRScheme
    # data
    TA::TensorMap
    TB::TensorMap

    finalize!::Function
    function LoopTNR(TA::TensorMap, TB::TensorMap; finalize=finalize!)
        return new(TA, TB, finalize)
    end
    function LoopTNR(T::TensorMap; finalize=finalize!)
        return new(T, copy(T), finalize)
    end
end

function Ψ_A(scheme::LoopTNR)
    psi = AbstractTensorMap[permute(scheme.TA, ((2,), (1, 3, 4))),
                            permute(scheme.TB, ((1,), (3, 4, 2))),
                            permute(scheme.TA, ((3,), (4, 2, 1))),
                            permute(scheme.TB, ((4,), (2, 1, 3)))]
    return psi
end

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
        new_L = new_L / maximum(new_L.data)

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
        new_R = new_R / maximum(new_R.data)

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

#Functions to construct Ψ_B

function one_loop_projector(phi::Array, pos::Int, trunc::TensorKit.TruncationScheme)
    L = id(space(phi[1])[1])
    R = id(space(phi[end])[4]')
    for i in 1:pos
        L = QR_L(L, phi[i])
    end
    for i in length(phi):-1:(pos + 1)
        R = QR_R(R, phi[i])
    end
    PR, PL = P_decomp(R, L, trunc)
    return PR, PL
end

function SVD12(T::AbstractTensorMap{E,S,1,3}, trunc::TensorKit.TruncationScheme) where {E,S}
    U, s, V, _ = tsvd(T, ((1, 2), (3, 4)); trunc=trunc)
    @tensor S1[-1; -2 -3] := U[-1 -2; 1] * sqrt(s)[1; -3]
    @tensor S2[-1; -2 -3] := sqrt(s)[-1; 1] * V[1; -2 -3]
    return S1, S2
end

function Ψ_B(scheme::LoopTNR, trunc::TensorKit.TruncationScheme)
    ΨA = Ψ_A(scheme)
    ΨB = []

    for i in 1:4
        s1, s2 = SVD12(ΨA[i], trunc)
        push!(ΨB, s1)
        push!(ΨB, s2)
    end

    ΨB_function(steps, data) = abs(data[end])
    criterion = maxiter(100) & convcrit(1e-12, ΨB_function)
    PR_list, PL_list = find_projectors(ΨB, criterion, trunc)

    ΨB_disentangled = []
    for i in 1:8
        @tensor B1[-1; -2 -3] := PL_list[i][-1; 1] * ΨB[i][1; -2 2] *
                                 PR_list[mod(i, 8) + 1][2; -3]
        push!(ΨB_disentangled, B1)
    end
    return ΨB_disentangled
end

#Entanglement Filtering 
entanglement_function(steps, data) = abs(data[end])
entanglement_criterion = maxiter(100) & convcrit(1e-15, entanglement_function)

loop_criterion = maxiter(50) & convcrit(1e-5, entanglement_function)

function entanglement_filtering!(scheme::LoopTNR, entanglement_criterion::stopcrit,
                                 trunc::TensorKit.TruncationScheme)
    ΨA = Ψ_A(scheme)
    PR_list, PL_list = find_projectors(ΨA, entanglement_criterion, trunc)

    TA = copy(scheme.TA)
    TB = copy(scheme.TB)

    @tensor scheme.TA[-1 -2; -3 -4] := TA[1 2; 3 4] * PR_list[4][1; -1] *
                                       PL_list[1][-2; 2] * PR_list[2][4; -4] *
                                       PL_list[3][-3; 3]
    @tensor scheme.TB[-1 -2; -3 -4] := TB[1 2; 3 4] * PL_list[2][-1; 1] *
                                       PR_list[3][2; -2] * PL_list[4][-4; 4] *
                                       PR_list[1][3; -3]

    return scheme
end

function entanglement_filtering!(scheme::LoopTNR, trunc::TensorKit.TruncationScheme)
    return entanglement_filtering!(scheme, entanglement_criterion, trunc)
end

#cost functions

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

#Optimisation functions

function tN(pos, psiB)
    @tensor tmp[-1 -2; -3 -4] := psiB[mod(pos, 8) + 1][-2; 1 -4] *
                                 conj(psiB[mod(pos, 8) + 1][-1 1; -3])
    for i in (pos + 1):(pos + 6)
        ΨB = psiB[mod(i, 8) + 1]
        @tensor tmp[-1 -2; -3 -4] := tmp[-1 -2; 1 2] * ΨB[2; 3 -4] * conj(ΨB[1; 3 -3])
    end
    return tmp
end

function tW(pos, psiA, psiB)
    next_a = mod(ceil(Int, pos / 2), 4) + 1
    next_b = mod(2 * ceil(Int, pos / 2) + 1, 8)

    @tensor tmp[-1 -2; -3 -4] := psiA[next_a][-2; 1 2 -4] * conj(psiB[next_b][-1; 1 3]) *
                                 conj(psiB[next_b + 1][3; 2 -3])
    for i in next_a:(next_a + 1)
        ΨA = psiA[mod(i, 4) + 1]
        ΨB1 = psiB[2 * (mod(i, 4) + 1) - 1]
        ΨB2 = psiB[2 * (mod(i, 4) + 1)]
        @tensor tmp[-1 -2; -3 -4] := tmp[-1 -2; 1 2] * ΨA[2; 3 4 -4] * conj(ΨB1[1; 3 5]) *
                                     conj(ΨB2[5; 4 -3])
    end

    if pos % 2 == 0
        ΨA = psiA[ceil(Int, pos / 2)]
        ΨB = psiB[pos - 1]
        @tensor W[-1; -2 -3] := tmp[-3 1; 2 3] * conj(ΨB[2 4; -1]) * ΨA[3; 4 -2 1]
    else
        ΨA = psiA[ceil(Int, pos / 2)]
        ΨB = psiB[pos + 1]
        @tensor W[-1; -2 -3] := tmp[1 2; -1 3] * ΨA[3; -2 4 2] * conj(ΨB[-3; 4 1])
    end

    return W
end

function opt_T(N, W, psi)
    function apply_f(x::TensorMap)
        @tensor b[-1; -2 -3] := N[-3 2; -1 1] * x[1; -2 2]
        return b
    end

    new_T, info = linsolve(apply_f, W, psi; krylovdim=50, maxiter=150, tol=1e-12,
                           verbosity=0)
    return new_T
end

function loop_opt!(scheme::LoopTNR, loop_criterion::stopcrit,
                   trunc::TensorKit.TruncationScheme, verbosity::Int)
    psi_A = Ψ_A(scheme)
    psi_B = Ψ_B(scheme, trunc)

    cost = ComplexF64[Inf]
    sweep = 0
    crit = true
    while crit
        for i in 1:8
            N = tN(i, psi_B)
            W = tW(i, psi_A, psi_B)
            new_T = opt_T(N, W, psi_B[i])
            psi_B[i] = new_T
        end
        sweep += 1
        push!(cost, cost_func(1, psi_A, psi_B))
        if verbosity > 1
            @infov 3 "Sweep: $sweep, Cost: $(cost[end])"
        end
        crit = loop_criterion(sweep, cost)
    end

    Ψ5 = psi_B[5]
    Ψ8 = psi_B[8]
    Ψ1 = psi_B[1]
    Ψ4 = psi_B[4]

    @tensor scheme.TA[-1 -2; -3 -4] := Ψ1[1; 2 -2] * Ψ4[-4; 2 3] * Ψ5[3; 4 -3] * Ψ8[-1; 4 1]

    Ψ2 = psi_B[2]
    Ψ3 = psi_B[3]
    Ψ6 = psi_B[6]
    Ψ7 = psi_B[7]

    @tensor scheme.TB[-1 -2; -3 -4] := Ψ6[-2; 1 2] * Ψ7[2; 3 -4] * Ψ2[-3; 3 4] * Ψ3[4; 1 -1]
    return scheme
end

function loop_opt!(scheme::LoopTNR, trunc::TensorKit.TruncationScheme,
                   verbosity::Int)
    return loop_opt!(scheme, loop_criterion, trunc, verbosity)
end

function my_inner(x, v1, v2)
    return real(dot(v1, v2))
end

function step!(scheme::LoopTNR, trunc::TensorKit.TruncationScheme,
               entanglement_criterion::stopcrit,
               loop_criterion::stopcrit, verbosity::Int)
    entanglement_filtering!(scheme, entanglement_criterion, trunc)
    loop_opt!(scheme, loop_criterion, trunc, verbosity::Int)
    return scheme
end

function run!(scheme::LoopTNR, trscheme::TensorKit.TruncationScheme, criterion::stopcrit,
              entanglement_criterion::stopcrit,
              loop_criterion::stopcrit;
              finalize_beginning=true, verbosity=1)
    data = []

    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting simulation\n $(scheme)\n"
        if finalize_beginning
            push!(data, scheme.finalize!(scheme))
        end

        steps = 0
        crit = true

        t = @elapsed while crit
            @infov 2 "Step $(steps + 1), data[end]: $(!isempty(data) ? data[end] : "empty")"
            step!(scheme, trscheme, entanglement_criterion, loop_criterion, verbosity)
            push!(data, scheme.finalize!(scheme))
            steps += 1
            crit = criterion(steps, data)
        end

        @infov 1 "Simulation finished\n $(stopping_info(criterion, steps, data))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return data
end

function run!(scheme::LoopTNR, trscheme::TensorKit.TruncationScheme, criterion::stopcrit;
              finalize_beginning=true, verbosity=1)
    return run!(scheme, trscheme, criterion, entanglement_criterion, loop_criterion;
                finalize_beginning=finalize_beginning,
                verbosity=verbosity)
end

function Base.show(io::IO, scheme::LoopTNR)
    println(io, "LoopTNR - Loop Tensor Network Renormalization")
    println(io, "  * TA: $(summary(scheme.TA))")
    println(io, "  * TB: $(summary(scheme.TB))")
    return nothing
end
