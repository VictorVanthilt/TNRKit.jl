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

"""
Tensors in a loop is initially 
    |       |
    v       v
    3       3
-1<-B-4-<-1-A-4<--
    2       2
    v       v
    3       3
-1<-A-4-<-1-B-4<--
    2       2
    v       v
    |       |
It is transformed into the tensor array
     |         |
     v         v
     3         2
-->2-4-4<---<1-1-3<--
     1         4
     v         ^
     4         1
-->3-3-1>--->4-2-2<--
     2         3
     ^         ^
     |         |
"""
function Ψ_A(scheme::LoopTNR)
    psi = AbstractTensorMap[transpose(scheme.TA, ((1,), (3, 4, 2))),
                            transpose(scheme.TB, ((3,), (4, 2, 1))),
                            transpose(scheme.TA, ((4,), (2, 1, 3))),
                            transpose(scheme.TB, ((2,), (1, 3, 4)))]
    return psi
end

#Utility functions for QR decomp

"""
      |     |
       2   3
        v v
--L-1-<--T--<-4-----
=
      |     |
       2   3
        v v
----1-<--Q--<-4--Rt-
"""
function QR_L(L::TensorMap, T::AbstractTensorMap{E,S,1,3}) where {E,S}
    @planar LT[-1; -2 -3 -4] := L[-1; 1] * T[1; -2 -3 -4]
    temp = transpose(LT, (3,2,1), (4,))
    _, Rt = leftorth(temp,)
    return Rt/norm(Rt, Inf)
end

"""
       |     |
        2   3
         v v
-----1-<--T--<-4-R--
=
       |     |
        2   3
         v v
-Lt--1-<--Q--<-4----
"""
function QR_R(R::TensorMap, T::AbstractTensorMap{E,S,1,3}) where {E,S}
    @planar TR[-1; -2 -3 -4] := T[-1; -2 -3 1] * R[1; -4]
    Lt, _ = rightorth(TR,)
    return Lt/norm(Lt, Inf)
end

"""
         |
         2
         v
--L-1-<--T--<-3----
=
         | 
         2
         v
----1-<--Q--<-3--Rt-
"""
function QR_L(L::TensorMap, T::AbstractTensorMap{E,S,1,2}) where {E,S}
    @planar LT[-1; -2 -3] := L[-1; 1] * T[1; -2 -3]
    temp = transpose(LT, (2, 1), (3,))
    _, Rt = leftorth(temp,)
    return Rt/norm(Rt, Inf)
end


"""
          |
          2
          v
-----1-<--T--<-3-R--
=
          |
          2
          v
-Lt--1-<--Q--<-3----
"""
function QR_R(R::TensorMap, T::AbstractTensorMap{E,S,1,2}) where {E,S}
    @planar TR[-1; -2 -3] := T[-1; -2 1] * R[1; -3]
    Lt, _ = rightorth(TR,)
    return Lt/norm(Lt, Inf)
end

#Functions to find the array of left and right projectors

function find_L(psi::Array, entanglement_criterion::stopcrit)
    type = eltype(psi[1])
    n = length(psi)
    L_list = map(pos->id(type, codomain(psi[pos])[1]), 1:n)
    crit = true
    steps = 0
    error = [Inf]
    while crit
        pos_next = mod(pos,n)+1
        L_last_time = L_list[pos_next]
        L_list[pos_next]= QR_L(L_list[pos], psi[pos])

        if space(L_list[pos_next]) == space(L_last_time)
            push!(error, abs(norm(L_list[pos_next] - L_last_time)))
        end

        steps += 1
        crit = entanglement_criterion(steps, error)
    end

    return L_list
end

function find_R(psi::Array, entanglement_criterion::stopcrit)
    type = eltype(psi[1])
    n = length(psi)
    R_list = map(pos->id(type, domain(psi[pos]).spaces[end]), 1:n)
    crit = true
    steps = 0
    error = [Inf]
    while crit
        pos_last = mod(pos-2,n)+1
        R_last_time = R_list[pos_last]
        R_list[pos_last] = QR_R(R_list[pos], psi[pos])

        if space(R_list[end]) == space(R_last_time)
            push!(error, abs(norm(R_list[end] - R_last_time)))
        end

        steps += 1
        crit = entanglement_criterion(steps, error)
    end

    return R_list
end

function P_decomp(R::TensorMap, L::TensorMap, trunc::TensorKit.TruncationScheme)
    @planar LR[-1; -2] := L[-1; 1] * R[1; -2]
    U, S, V, _ = tsvd(LR; trunc=trunc)

    @planar PR[-1; -2] := R[-1; 1] * V'[1; 2] * inv(sqrt(S))[2; -2]
    @planar PL[-1; -2] := inv(sqrt(S))[-1; 1] * U'[1; 2] * L[2; -2]

    return PR, PL
end

function find_projectors(psi::Array, entanglement_criterion::stopcrit,
                         trunc::TensorKit.TruncationScheme)
    n = length(psi)
    PR_list = Vector(undef, n)
    PL_list = Vector(undef, n)
    L_list = find_L(psi, entanglement_criterion)
    R_list = find_R(psi, entanglement_criterion)


    for i in 1:n
        PR_list[mod(i-2,n)+1], PL_list[i] = P_decomp(R_list[mod(i-2,4)+1], L_list[i], trunc)
    end
    return PR_list, PL_list
end

#Functions to construct Ψ_B

function one_loop_projector(phi::Array, pos::Int, trunc::TensorKit.TruncationScheme)
    L = id(codomain(phi[1])[1])
    R = id(domain(phi[end]).spaces[end])
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
    T_trans = transpose(T, (2,1), (3,4))
    U, s, V, _ = tsvd(T_trans; trunc=trunc)
    @planar S1[-1; -2 -3] := U[-2 -1; 1] * sqrt(s)[1; -3]
    @planar S2[-1; -2 -3] := sqrt(s)[-1; 1] * V[1; -2 -3]
    return S1, S2
end

function Ψ_B(scheme::LoopTNR, trunc::TensorKit.TruncationScheme)
    ΨA = Ψ_A(scheme)
    ΨB = []

    for i in 1:4
        s1, s2 = SVD12(ΨA[i], truncdim(trunc.dim * 2))
        push!(ΨB, s1)
        push!(ΨB, s2)
    end

    ΨB_function(steps, data) = abs(data[end])
    criterion = maxiter(100) & convcrit(1e-12, ΨB_function)
    PR_list, PL_list = find_projectors(ΨB, criterion, trunc)

    for i in 1:8
        @planar B1[-1; -2 -3] := PL_list[i][-1; 1] * ΨB[i][1; -2 2] *
                                 PR_list[mod(i, 8) + 1][2; -3]
        ΨB[i] = B1
    end
    return ΨB
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

    @planar scheme.TA[-1 -2; -3 -4] := TA[1 2; 3 4] * PL_list[1][-1; 1] * PR_list[1][2; -2] * PR_list[3][3; -3] * PL_list[3][-4; 4]
    @planar scheme.TB[-1 -2; -3 -4] := TB[1 2; 3 4] * PR_list[2][1; -1] *
                                       PL_list[4][-2; 2] * PL_list[2][-3; 3] *
                                       PR_list[4][4; -4]

    return scheme
end

function entanglement_filtering!(scheme::LoopTNR, trunc::TensorKit.TruncationScheme)
    return entanglement_filtering!(scheme, entanglement_criterion, trunc)
end
#cost functions

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
        push!(ΨAΨA_list, temp)
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
        @planar temp[-1 -2; -3 -4] := psiB[2*i-1]'[-1; 1 3] * psiB[2*i]'[3; 2 -3] * psiA[i][-2; 1 2 -4]
        push!(ΨBΨA_list, temp)
    end
    return ΨBΨA_list
end

function to_number(tensor_list)
    cont = tensor_list[1]
    for i in 2:length(loop_array)
        cont = cont * tensor_list[i]
    end

    @planar num = cont[d u; d u]
    return num
end

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
    BB = psiBpsiB[next_pos]
    for i in 2:(n-1)
        pos = mod(pos, 8) + 1
        BB = BB * psiBpsiB[pos]
    end
    return BB
end

function tW(pos, psiA, psiB, psiBpsiA)
    pos_psiA = (pos-1)÷2+1
    ΨA = psiA[pos_psiA]

    next_a = mod(pos_psiA, 4) + 1
    tmp = psiBpsiA[next_a]
    for i in 1:2
        site = mod(next_a,4) + 1
        tmp = tmp * psiBpsiA[site]
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
        @planar W[-1; -2 -3] := ΨB'[4 -1; 2] * ΨA[3; 4 -3 1] * tmp[-2 1; 2 3]
    else
        ΨB = psiB[pos + 1]
        #-1'--   --2'--ΨB--2----------1'-
        #      |       |         |
        #      3'      4         t
        #       |     |          m
        #        |   |           p
        #         | |            |
        #---3------ΨA------1----------3--
        @planar W[-1; -2 -3] := ΨB'[2 4; -2] * ΨA[3; -3 4 1] * tmp[2 1; -1 3]
    end

    return W
end

function opt_T(N, W, psi)
    function apply_f(x::TensorMap)
        @tensor b[-1; -2 -3] := N[-2 2; -1 1] * x[1; -3 2]
        return b
    end
    new_T, info = linsolve(apply_f, W, psi; krylovdim=10, maxiter=100, tol=1e-10, verbosity=0)
    return new_T
end

function loop_opt!(scheme::LoopTNR, loop_criterion::stopcrit,
                   trunc::TensorKit.TruncationScheme, verbosity::Int)
    psiA = Ψ_A(scheme)
    psiB = Ψ_B(scheme, trunc)
    psiBpsiB = ΨBΨB(psiB)
    psiBpsiA = ΨBΨA(psiB, psiA)
    psiApsiA = ΨAΨA(psiA)

    cost = ComplexF64[Inf]
    sweep = 0
    crit = true
    while crit
        for i in 1:8
            N = tN(i, psiBpsiB)
            W = tW(i, psiA, psiB, psiBpsiA)
            new_S = opt_T(N, W, psiB[i])
            psiB[i] = new_S

            @planar SS[-1 -2; -3 -4] := new_S[-2; 1 -4] * new_S'[1 -3; -1]
            psiBpsiB[i] = SS

            pos_psiA = (i-1)÷2+1
            @planar TSS[-1 -2; -3 -4] := psiB[2*pos_psiA-1]'[-1; 1 3] * psiB[2*pos_psiA]'[3; 2 -3] * psiA[pos_psiA][-2; 1 2 -4]
            psiBpsiA[pos_psiA] = TSS
        end
        sweep += 1
        push!(cost, cost_func(psiApsiA, psiBpsiB, psiBpsiA))
        if verbosity > 1
            @infov 3 "Sweep: $sweep, Cost: $(cost[end])"
        end
        crit = loop_criterion(sweep, cost)
    end

    Ψ5 = psiB[5]
    Ψ8 = psiB[8]
    Ψ1 = psiB[1]
    Ψ4 = psiB[4]

    @planar scheme.TB[-1 -2; -3 -4] := Ψ1[1; 2 -2] * Ψ4[-4; 2 3] * Ψ5[3; 4 -3] * Ψ8[-1; 4 1]

    Ψ2 = psiB[2]
    Ψ3 = psiB[3]
    Ψ6 = psiB[6]
    Ψ7 = psiB[7]

    @planar scheme.TA[-1 -2; -3 -4] := Ψ6[-2; 1 2] * Ψ7[2; 3 -4] * Ψ2[-3; 3 4] * Ψ3[4; 1 -1]
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
