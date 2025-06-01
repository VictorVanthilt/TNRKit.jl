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

# Function to initialize the list of tensors Ψ_A, making it an MPS on a ring
function Ψ_A(scheme::LoopTNR)
    psi = AbstractTensorMap[transpose(scheme.TA, ((2,), (1, 3, 4))),
                            transpose(scheme.TB, ((1,), (3, 4, 2))),
                            transpose(scheme.TA, ((3,), (4, 2, 1))),
                            transpose(scheme.TB, ((4,), (2, 1, 3)))]
    return psi
end

#Utility functions for QR decomp

# A single step of the QR decomposition from the left with 3 in-coming legs
#       |     |
#        2   3
#         v v
# --L-1-<--T--<-4-----
# =
#       |     |
#        2   3
#         v v
# ----1-<--Q--<-4--Rt-

function QR_L(L::TensorMap, T::AbstractTensorMap{E,S,1,3}) where {E,S}
    @planar LT[-1; -2 -3 -4] := L[-1; 1] * T[1; -2 -3 -4]
    temp = transpose(LT, (3, 2, 1), (4,))
    _, Rt = leftorth(temp)
    return Rt/norm(Rt, Inf)
end

# A single step of the QR decomposition from the right with 3 in-coming legs
#        |     |
#         2   3
#          v v
# -----1-<--T--<-4-R--
# =
#        |     |
#         2   3
#          v v
# -Lt--1-<--Q--<-4----

function QR_R(R::TensorMap, T::AbstractTensorMap{E,S,1,3}) where {E,S}
    @planar TR[-1; -2 -3 -4] := T[-1; -2 -3 1] * R[1; -4]
    Lt, _ = rightorth(TR)
    return Lt/norm(Lt, Inf)
end

# A single step of the QR decomposition from the left with 2 in-coming legs
#          |
#          2
#          v
# --L-1-<--T--<-3----
# =
#          | 
#          2
#          v
# ----1-<--Q--<-3--Rt-

function QR_L(L::TensorMap, T::AbstractTensorMap{E,S,1,2}) where {E,S}
    @planar LT[-1; -2 -3] := L[-1; 1] * T[1; -2 -3]
    temp = transpose(LT, (2, 1), (3,))
    _, Rt = leftorth(temp)
    return Rt/norm(Rt, Inf)
end

# A single step of the QR decomposition from the right with 2 in-coming legs
#           |
#           2
#           v
# -----1-<--T--<-3-R--
# =
#           |
#           2
#           v
# -Lt--1-<--Q--<-3----
function QR_R(R::TensorMap, T::AbstractTensorMap{E,S,1,2}) where {E,S}
    @planar TR[-1; -2 -3] := T[-1; -2 1] * R[1; -3]
    Lt, _ = rightorth(TR)
    return Lt/norm(Lt, Inf)
end

# Functions to find the left and right projectors

# Function to find the list of left projectors L_list
function find_L(psi::Array, entanglement_criterion::stopcrit)
    type = eltype(psi[1])
    n = length(psi)
    L_list = map(x->id(type, codomain(psi[x])[1]), 1:n)
    crit = true
    steps = 0
    error = [Inf]
    running_pos = 1
    while crit
        pos_next = mod(running_pos, n) + 1
        L_last_time = L_list[pos_next]
        L_list[pos_next] = QR_L(L_list[running_pos], psi[running_pos])

        if space(L_list[pos_next]) == space(L_last_time)
            push!(error, abs(norm(L_list[pos_next] - L_last_time)))
        end

        running_pos = pos_next
        steps += 1
        crit = entanglement_criterion(steps, error)
    end

    return L_list
end

# Function to find the list of right projectors R_list
function find_R(psi::Array, entanglement_criterion::stopcrit)
    type = eltype(psi[1])
    n = length(psi)
    R_list = map(x->id(type, domain(psi[x]).spaces[end]), 1:n)
    crit = true
    steps = 0
    error = [Inf]

    running_pos = n
    while crit
        pos_last = mod(running_pos-2, n)+1
        R_last_time = R_list[pos_last]
        R_list[pos_last] = QR_R(R_list[running_pos], psi[running_pos])

        if space(R_list[pos_last]) == space(R_last_time)
            push!(error, abs(norm(R_list[pos_last] - R_last_time)))
        end

        running_pos = pos_last
        steps += 1
        crit = entanglement_criterion(steps, error)
    end

    return R_list
end

# Function to find the projector P_L and P_R
function P_decomp(R::TensorMap, L::TensorMap, trunc::TensorKit.TruncationScheme)
    U, S, V, _ = tsvd(L * R; trunc=trunc)

    PR = R * V' * inv(sqrt(S))
    PL = inv(sqrt(S)) * U' * L

    return PR, PL
end

# Function to find the list of projectors
function find_projectors(psi::Array, entanglement_criterion::stopcrit,
                         trunc::TensorKit.TruncationScheme)
    PR_list = []
    PL_list = []

    n = length(psi)
    L_list = find_L(psi, entanglement_criterion)
    R_list = find_R(psi, entanglement_criterion)
    for i in 1:n
        pr, pl = P_decomp(R_list[mod(i-2, n)+1], L_list[i], trunc)

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
    T_trans = transpose(T, (2, 1), (3, 4))
    U, s, V, _ = tsvd(T_trans; trunc=trunc, alg=TensorKit.SVD())
    @planar S1[-1; -2 -3] := U[-2 -1; 1] * sqrt(s)[1; -3]
    @planar S2[-1; -2 -3] := sqrt(s)[-1; 1] * V[1; -2 -3]
    return S1, S2
end

# Function to construct MPS Ψ_B from MPS Ψ_A. Using a large cut-off dimension in SVD but a small cut-off dimension in loop to increase the precision of initialization.
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
        @planar B1[-1; -2 -3] := PL_list[i][-1; 1] * ΨB[i][1; -2 2] *
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

# Construct the list of transfer matrices for ΨAΨA
# ---1'--A--3'---
#       | |
#       1 2
#       | |
# ---2'--A--4'---
function ΨAΨA(psiA)
    ΨAΨA_list = []
    for i in 1:4
        @planar tmp[-1 -2; -3 -4] := psiA[i][-2; 1 2 -4] * psiA[i]'[1 2 -3; -1]
        push!(ΨAΨA_list, tmp)
    end
    return ΨAΨA_list
end

# Construct the list of transfer matrices for ΨBΨB
# ---1'--B--3'---
#        |
#        1
#        |
# ---2'--B--4'---

function ΨBΨB(psiB)
    ΨBΨB_list = []
    for i in 1:8
        @planar tmp[-1 -2; -3 -4] := psiB[i][-2; 1 -4] * psiB[i]'[1 -3; -1]
        push!(ΨBΨB_list, tmp)
    end
    return ΨBΨB_list
end

# Construct the list of transfer matrices for ΨBΨA
# ---1'--B-3-B--3'---
#        |   |
#        1   2
#         | |
# ---2'----A----4'---

function ΨBΨA(psiB, psiA)
    ΨBΨA_list = []
    for i in 1:4
        @planar temp[-1 -2; -3 -4] := psiB[2*i-1]'[1 3; -1] * psiA[i][-2; 1 2 -4] *
                                      psiB[2*i]'[2 -3; 3]
        push!(ΨBΨA_list, temp)
    end
    return ΨBΨA_list
end

# Function to compute the trace of a list of transfer matrices
function to_number(tensor_list)
    cont = tensor_list[1]
    for tensor in tensor_list[2:end]
        cont = cont * tensor
    end
    return tr(cont)
end

#Entanglement Filtering 
entanglement_function(steps, data) = abs(data[end])
entanglement_criterion = maxiter(100) & convcrit(1e-15, entanglement_function)

loop_criterion = maxiter(50) & convcrit(1e-8, entanglement_function)

# Entanglement filtering function
function entanglement_filtering!(scheme::LoopTNR, entanglement_criterion::stopcrit,
                                 trunc::TensorKit.TruncationScheme)
    ΨA = Ψ_A(scheme)
    PR_list, PL_list = find_projectors(ΨA, entanglement_criterion, trunc)

    TA = copy(scheme.TA)
    TB = copy(scheme.TB)

    @planar scheme.TA[-1 -2; -3 -4] := TA[1 2; 3 4] * PR_list[4][1; -1] *
                                       PL_list[1][-2; 2] * PR_list[2][4; -4] *
                                       PL_list[3][-3; 3]
    @planar scheme.TB[-1 -2; -3 -4] := TB[1 2; 3 4] * PL_list[2][-1; 1] *
                                       PR_list[3][2; -2] * PL_list[4][-4; 4] *
                                       PR_list[1][3; -3]

    return scheme
end

function entanglement_filtering!(scheme::LoopTNR, trunc::TensorKit.TruncationScheme)
    return entanglement_filtering!(scheme, entanglement_criterion, trunc)
end

#Optimisation functions

# Function to compute the half of the matrix N by inputting the left and right SS transfer matrices
tN(SS_left, SS_right) = SS_right * SS_left

# Function to compute the vector W for a given position in the loop
function tW(pos, psiA, psiB, TSS_left, TSS_right)
    ΨA = psiA[(pos-1)÷2+1]

    tmp = TSS_right * TSS_left

    if iseven(pos)
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

# Function to optimize the tensor T for a given position in the loop by the Krylov method
function opt_T(N, W, psi)
    function apply_f(x::TensorMap)
        #-----1'--   --2'------------1'--
        #          |            |
        #          3'           |
        #          |            N
        #          |            |
        #          |            |
        #---1------x-------2---------1---
        @planar b[-1; -3 -2] := N[-2 2; -1 1] * x[1; -3 2]
        return b
    end
    new_T, info = linsolve(apply_f, W, psi; krylovdim=20, maxiter=20, tol=1e-12,
                           verbosity=0)
    return new_T
end

# Function to compute the right cache for the transfer matrices. Here we sweep from left to right. At the end we add the identity transfer matrix to the cache.
# cache[1] = T2 * T3 * T4 * T5 * T6 * T7 * T8
# cache[2] = T3 * T4 * T5 * T6 * T7 * T8
# cache[3] = T4 * T5 * T6 * T7 * T8
# ...
# cache[7] = T8
# cache[8] = I
function right_cache(tensor_list)
    n = length(tensor_list)
    cache = similar(tensor_list)
    cache[end] = id(domain(tensor_list[end]))

    for i in (n - 1):-1:1
        cache[i] = tensor_list[i + 1] * cache[i + 1]
    end

    return cache
end

# Function to perform the optimization loop for the LoopTNR scheme. Sweeping from left to right, we optimize the tensors in the loop by minimizing the cost function.
# Here cache of right-half-chain is used to minimize the number of multiplications to accelerate the sweeping. 
# The transfer matrix on the left is updated after each optimization step.
function loop_opt!(scheme::LoopTNR, loop_criterion::stopcrit,
                   trunc::TensorKit.TruncationScheme, verbosity::Int)
    psiA = Ψ_A(scheme)
    psiB = Ψ_B(psiA, trunc)
    psiBpsiB = ΨBΨB(psiB)
    psiBpsiA = ΨBΨA(psiB, psiA)
    psiApsiA = ΨAΨA(psiA)
    C = to_number(psiApsiA) # Since C is not changed during the optimization, we can compute it once and use it in the cost function.

    cost = ComplexF64[Inf]
    sweep = 0
    crit = true
    while crit
        right_cache_BB = right_cache(psiBpsiB)
        right_cache_BA = right_cache(psiBpsiA)
        left_BB = id(codomain(psiBpsiB[1])) # Initialize the left transfer matrix for ΨBΨB
        left_BA = id(codomain(psiBpsiA[1])) # Initialize the left transfer matrix for ΨBΨA

        t_start = time()
        for pos_psiB in 1:8
            pos_psiA = (pos_psiB-1)÷2+1 # Position in the MPS Ψ_A

            N = tN(left_BB, right_cache_BB[pos_psiB]) # Compute the half of the matrix N for the current position in the loop, right cache is used to minimize the number of multiplications
            W = tW(pos_psiB, psiA, psiB, left_BA, right_cache_BA[pos_psiA]) # Compute the vector W for the current position in the loop, using the right cache for ΨBΨA

            new_psiB = opt_T(N, W, psiB[pos_psiB]) # Optimize the tensor T for the current position in the loop, with the psiB[pos_psiB] be the initial guess

            psiB[pos_psiB] = new_psiB # Update a single local tensor in the MPS Ψ_B 

            @planar BB_temp[-1 -2; -3 -4] := new_psiB[-2; 1 -4] * new_psiB'[1 -3; -1]
            psiBpsiB[pos_psiB] = BB_temp # Update the transfer matrix for ΨBΨB
            left_BB = left_BB * BB_temp # Update the left transfer matrix for ΨBΨB

            if iseven(pos_psiB) # If the position is even, we also update the transfer matrix for ΨBΨA
                @planar BA_temp[-1 -2; -3 -4] := psiB[2*pos_psiA-1]'[1 3; -1] *
                                                 psiA[pos_psiA][-2; 1 2 -4] *
                                                 psiB[2*pos_psiA]'[2 -3; 3]
                psiBpsiA[pos_psiA] = BA_temp # Update the transfer matrix for ΨBΨA
                left_BA = left_BA * BA_temp # Update the left transfer matrix for ΨBΨA
            end
        end
        sweep += 1
        crit = loop_criterion(sweep, cost)

        tNt = tr(left_BB)
        tdw = tr(left_BA)
        wdt = conj(tdw)
        cost_this = (C + tNt - wdt - tdw)/C
        push!(cost, cost_this)

        if verbosity > 1
            @infov 3 "Sweep: $sweep, Cost: $(cost[end]), Time: $(time() - t_start)s" # Included the time taken for the sweep
        end
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
