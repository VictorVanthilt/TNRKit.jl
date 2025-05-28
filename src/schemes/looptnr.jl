#TODO: Add documentation
mutable struct LoopTNR <: TNRScheme
    TA::TensorMap
    TB::TensorMap

    finalize!::Function
    function LoopTNR(TA::TensorMap, TB::TensorMap; finalize=(finalize!))
        return new(TA, TB, finalize)
    end
    function LoopTNR(T::TensorMap; finalize=(finalize!))
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

#Functions to construct Ψ_B

function one_loop_projector(phi::Array, pos::Int, trunc::TensorKit.TruncationScheme)
    L = id(space(phi[1])[1])
    n = length(phi)
    if numin(phi[n]) == 2
        R = id(space(phi[n])[3]')
    else
        R = id(space(phi[n])[4]')
    end
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

function Ψ_B(ΨA, trunc::TensorKit.TruncationScheme)
    ΨB = []

    for i in 1:4
        s1, s2 = SVD12(ΨA[i], truncdim(trunc.dim * 2))
        push!(ΨB, s1)
        push!(ΨB, s2)
    end

    ΨB_function(steps, data) = abs(data[end])
    criterion = maxiter(10) & convcrit(1e-12, ΨB_function)
    PR_list, PL_list = find_projectors(ΨB, criterion, trunc)

    ΨB_disentangled = []
    for i in 1:8
        @tensor B1[-1; -2 -3] := PL_list[i][-1; 1] * ΨB[i][1; -2 2] *
                                 PR_list[mod(i, 8) + 1][2; -3]
        push!(ΨB_disentangled, B1)
    end
    return ΨB_disentangled
end

function Ψ_B_oneloop(scheme::LoopTNR, trunc::TensorKit.TruncationScheme)
    ΨA = Ψ_A(scheme)
    ΨB = []

    for i in 1:4
        s1, s2 = SVD12(ΨA[i], truncdim(trunc.dim * 2))
        phi = copy(ΨA)
        deleteat!(phi, i)
        insert!(phi, i, s1)
        insert!(phi, i + 1, s2)
        PR, PL = one_loop_projector(phi, i, trunc)
        @tensor B1[-1; -2 -3] := s1[-1; -2 1] * PR[1; -3]
        @tensor B2[-1; -2 -3] := PL[-1; 1] * s2[1; -2 -3]
        push!(ΨB, B1)
        push!(ΨB, B2)
    end
    return ΨB
end

"""
---1'--A--3'---
      | |
      1 2
      | |
---2'--A--4'---
"""
function ΨAΨA(psiA)
    ΨAΨA_list = []
    for i in 1:4
        @planar tmp[-1 -2; -3 -4] := psiA[i][-2; 1 2 -4] * psiA[i]'[1 2 -3; -1]
        push!(ΨAΨA_list, tmp)
    end
    return ΨAΨA_list
end

"""
---1'--B--3'---
       |
       1
       |
---2'--B--4'---
"""
function ΨBΨB(psiB)
    ΨBΨB_list = []
    for i in 1:8
        @planar tmp[-1 -2; -3 -4] := psiB[i][-2; 1 -4] * psiB[i]'[1 -3; -1]
        push!(ΨBΨB_list, tmp)
    end
    return ΨBΨB_list
end

"""
---1'--B-3-B--3'---
       |   |
       1   2
        | |
---2'----A----4'---
"""
function ΨBΨA(psiB, psiA)
    ΨBΨA_list = []
    for i in 1:4
        @planar temp[-1 -2; -3 -4] := psiB[2*i-1]'[1 3; -1] * psiA[i][-2; 1 2 -4] * psiB[2*i]'[2 -3; 3]
        push!(ΨBΨA_list, temp)
    end
    return ΨBΨA_list
end

function to_number(tensor_list)
    cont = tensor_list[1]
    for tensor in tensor_list[2:end]
        cont = cont * tensor
    end
    return tr(cont)
end

function cost_func(psiApsiA, psiBpsiB, psiBpsiA)
    C = to_number(psiApsiA)
    tNt = to_number(psiBpsiB)
    tdw = to_number(psiBpsiA)
    wdt = conj(tdw)

    return (C + tNt - wdt - tdw)/C
end

#Entanglement Filtering 
entanglement_function(steps, data) = abs(data[end])
entanglement_criterion = maxiter(100) & convcrit(1e-15, entanglement_function)

loop_criterion = maxiter(50) & convcrit(1e-8, entanglement_function)

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


function cost_func(psiApsiA, psiBpsiB, psiBpsiA)
    C = to_number(psiApsiA)
    tNt = to_number(psiBpsiB)
    tdw = to_number(psiBpsiA)
    wdt = conj(tdw)

    return (C + tNt - wdt - tdw)/C
end

#Optimisation functions
function tN(pos, psiBpsiB)
    n = length(psiBpsiB)
    pos = mod(pos, n) + 1
    BB = psiBpsiB[pos]
    for i in 2:(n-1)
        pos = mod(pos, 8) + 1
        BB = BB * psiBpsiB[pos]
    end
    return BB
end

function tW(pos, psiA, psiB, psiBpsiA)
    pos_psiA = (pos-1)÷2+1
    ΨA = psiA[pos_psiA]

    T_site = mod(pos_psiA, 4) + 1
    tmp = psiBpsiA[T_site]
    for i in 1:2
        T_site = mod(T_site,4) + 1
        tmp = tmp * psiBpsiA[T_site]
    end

    if pos % 2 == 0
        ΨB = psiB[pos - 1]
        #--2---ΨB--1'-   --2'---------2--
        #      |       |         |
        #      4       3'        t
        #       |     |          m
        #        |   |           p
        #         | |            |
        #---3------ΨA------1----------3--
        @planar W[-1; -3 -2] := ΨB'[4 -1; 2] * ΨA[3; 4 -3 1] * tmp[-2 1; 2 3]
    else
        ΨB = psiB[pos + 1]
        #-1'--   --2'--ΨB--2----------1'-
        #      |       |         |
        #      3'      4         t
        #       |     |          m
        #        |   |           p
        #         | |            |
        #---3------ΨA------1----------3--
        @planar W[-1; -3 -2] := ΨB'[4 2; -2] * ΨA[3; -3 4 1] * tmp[2 1; -1 3]
    end

    return W
end

function opt_T(N, W, psi)
    function apply_f(x::TensorMap)
        #-----1'--   --2'------------1'--
        #          |            |
        #          3'           |
        #          |            N
        #          |            |
        #          |            |
        #---1------x-------2---------1---
        @tensor b[-1; -3 -2] := N[-2 2; -1 1] * x[1; -3 2]
        return b
    end
    new_T, info = linsolve(apply_f, W, psi; krylovdim=10, maxiter=100, tol=1e-10, verbosity=0)
    return new_T
end

function loop_opt!(scheme::LoopTNR, loop_criterion::stopcrit,
                   trunc::TensorKit.TruncationScheme, verbosity::Int)
    psiA = Ψ_A(scheme)
    psiB = Ψ_B(psiA, trunc)
    psiBpsiB = ΨBΨB(psiB)
    psiBpsiA = ΨBΨA(psiB, psiA)
    psiApsiA = ΨAΨA(psiA)

    cost = ComplexF64[Inf]
    sweep = 0
    crit = true
    while crit
        push!(cost, cost_func(psiApsiA, psiBpsiB, psiBpsiA))
        if verbosity > 1
            @infov 3 "Sweep: $sweep, Cost: $(cost[end])"
        end
        for i in 1:8
            N = tN(i, psiBpsiB)
            W = tW(i, psiA, psiB, psiBpsiA)
            new_S = opt_T(N, W, psiB[i])
            psiB[i] = new_S

            @planar SS[-1 -2; -3 -4] := new_S[-2; 1 -4] * new_S'[1 -3; -1]
            psiBpsiB[i] = SS

            pos_psiA = (i-1)÷2+1
            @planar TSS[-1 -2; -3 -4] := psiB[2*pos_psiA-1]'[1 3; -1] * psiA[pos_psiA][-2; 1 2 -4] * psiB[2*pos_psiA]'[2 -3; 3]
            psiBpsiA[pos_psiA] = TSS
        end
        sweep += 1
        crit = loop_criterion(sweep, cost)
    end

    Ψ5 = psi_B[5]
    Ψ8 = psi_B[8]
    Ψ1 = psi_B[1]
    Ψ4 = psi_B[4]

    @tensor scheme.TB[-1 -2; -3 -4] := Ψ1[1; 2 -2] * Ψ4[-4; 2 3] * Ψ5[3; 4 -3] * Ψ8[-1; 4 1]

    Ψ2 = psi_B[2]
    Ψ3 = psi_B[3]
    Ψ6 = psi_B[6]
    Ψ7 = psi_B[7]

    @tensor scheme.TA[-1 -2; -3 -4] := Ψ6[-2; 1 2] * Ψ7[2; 3 -4] * Ψ2[-3; 3 4] * Ψ3[4; 1 -1]
    return scheme
end

function loop_opt!(scheme::LoopTNR, trunc::TensorKit.TruncationScheme,
                   verbosity::Int)
    return loop_opt!(scheme, loop_criterion, trunc, verbosity)
end

function step!(scheme::LoopTNR, trunc::TensorKit.TruncationScheme,
               truncentanglement::TensorKit.TruncationScheme,
               entanglement_criterion::stopcrit,
               loop_criterion::stopcrit, verbosity::Int)
    entanglement_filtering!(scheme, entanglement_criterion, truncentanglement)
    loop_opt!(scheme, loop_criterion, trunc, verbosity::Int)
    return scheme
end

function run!(scheme::LoopTNR, trscheme::TensorKit.TruncationScheme,
              truncentanglement::TensorKit.TruncationScheme, criterion::stopcrit,
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
            step!(scheme, trscheme, truncentanglement, entanglement_criterion,
                  loop_criterion, verbosity)
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
    return run!(scheme, trscheme, truncbelow(1e-15), criterion, entanglement_criterion,
                loop_criterion;
                finalize_beginning=finalize_beginning,
                verbosity=verbosity)
end

function Base.show(io::IO, scheme::LoopTNR)
    println(io, "LoopTNR - Loop Tensor Network Renormalization")
    println(io, "  * TA: $(summary(scheme.TA))")
    println(io, "  * TB: $(summary(scheme.TB))")
    return nothing
end
