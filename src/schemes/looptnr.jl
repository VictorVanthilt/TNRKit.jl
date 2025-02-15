
#TODO: Make this work for Fermions
#TODO: Add documentation
mutable struct Loop_TNR <: TNRScheme
    # data
    TA::TensorMap
    TB::TensorMap

    finalize!::Function
    function Loop_TNR(TA::TensorMap, TB::TensorMap; finalize=finalize!)
        return new(TA, TB, finalize)
    end
end

function Ψ_A(scheme::Loop_TNR)
    psi = AbstractTensorMap[permute(scheme.TA, (2,), (3, 4, 1)),
                            permute(scheme.TB, (1,), (2, 3, 4)),
                            permute(scheme.TA, (4,), (1, 2, 3)),
                            permute(scheme.TB, (3,), (4, 1, 2))]
    return psi
end

#Utility functions for QR decomp

function QR_L(L::TensorMap, T::AbstractTensorMap{E,S,2,2}) where {E,S}
    @tensor temp[-1 -2; -3 -4] := L[-2; 1] * T[-1 1; -3 -4]
    _, Rt = leftorth(temp, (1, 2, 4), (3,))
    return Rt
end

function QR_R(R::TensorMap, T::AbstractTensorMap{E,S,2,2}) where {E,S}
    @tensor temp[-1 -2; -3 -4] := T[-1 -2; 1 -4] * R[1; -3]
    Lt, _ = rightorth(temp, (2,), (1, 3, 4))
    return Lt
end

function QR_L(L::TensorMap, T::AbstractTensorMap{E,S,1,3}) where {E,S}
    @tensor temp[-1; -2 -3 -4] := L[-1; 1] * T[1; -2 -3 -4]
    _, Rt = leftorth(temp, (1, 3, 4), (2,))
    return Rt
end

function QR_R(R::TensorMap, T::AbstractTensorMap{E,S,1,3}) where {E,S}
    @tensor temp[-1; -2 -3 -4] := T[-1; 1 -3 -4] * R[1; -2]
    Lt, _ = rightorth(temp, (1,), (2, 3, 4))
    return Lt
end

function QR_L(L::TensorMap, T::AbstractTensorMap{E,S,1,2}) where {E,S}
    @tensor temp[-1; -2 -3] := L[-1; 1] * T[1; -2 -3]
    _, Rt = leftorth(temp, (1, 3), (2,))
    return Rt
end

function QR_R(R::TensorMap, T::AbstractTensorMap{E,S,1,2}) where {E,S}
    @tensor temp[-1; -2 -3] := T[-1; 1 -3] * R[1; -2]
    Lt, _ = rightorth(temp, (1,), (2, 3))
    return Lt
end

#Maximum function that works for any TensorMap
function maximumer(T::TensorMap)
    maxi = []
    for (_, d) in blocks(T)
        push!(maxi, maximum(abs.(d)))
    end
    return maximum(maxi)
end

#Functions to find the left and right projectors

function find_L(pos::Int, psi::Array, maxsteps::Int, minerror::Float64)
    L = id(space(psi[pos])[1])
    crit = true
    steps = 0
    error = Inf
    n = length(psi)
    while crit
        new_L = copy(L)
        for i in (pos - 1):(pos + n - 2)
            new_L = QR_L(new_L, psi[i % n + 1])
        end
        new_L = new_L / maximumer(new_L)

        if space(new_L) == space(L)
            error = abs(norm(new_L - L))
        end

        L = new_L
        steps += 1
        crit = steps < maxsteps && error > minerror
    end

    return L
end

function find_R(pos::Int, psi::Array, maxsteps::Int, minerror::Float64)
    R = id(space(psi[mod(pos - 2, 4) + 1])[2]')
    crit = true
    steps = 0
    error = Inf
    n = length(psi)
    while crit
        new_R = copy(R)

        for i in (pos - 2):-1:(pos - n - 1)
            new_R = QR_R(new_R, psi[mod(i, n) + 1])
        end
        new_R = new_R / maximumer(new_R)

        if space(new_R) == space(R)
            error = abs(norm(new_R - R))
        end
        R = new_R
        steps += 1
        crit = steps < maxsteps && error > minerror
    end

    return R
end

function P_decomp(R::TensorMap, L::TensorMap, trunc::TensorKit.TruncationScheme)
    @tensor temp[-1; -2] := L[-1; 1] * R[1; -2]
    U, S, V, _ = tsvd(temp, (1,), (2,); trunc=trunc)
    re_sq = pseudopow(S, -0.5)

    @tensor PR[-1; -2] := R[-1; 1] * adjoint(V)[1; 2] * re_sq[2; -2]
    @tensor PL[-1; -2] := re_sq[-1; 1] * adjoint(U)[1; 2] * L[2; -2]

    return PR, PL
end

function find_projectors(psi::Array, maxsteps::Int, minerror::Float64,
                         trunc::TensorKit.TruncationScheme)
    PR_list = []
    PL_list = []
    n = length(psi)
    for i in 1:n
        L = find_L(i, psi, maxsteps, minerror)

        R = find_R(i, psi, maxsteps, minerror)

        pr, pl = P_decomp(R, L, trunc)

        push!(PR_list, pr)
        push!(PL_list, pl)
    end
    return PR_list, PL_list
end

#Functions to construct Ψ_B

function one_loop_projector(phi::Array, pos::Int, trunc::TensorKit.TruncationScheme)
    L = id(space(phi[1])[1])
    R = id(space(phi[end])[2]')
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
    U, s, V, _ = tsvd(T, (1, 4), (2, 3); trunc=trunc)
    @tensor S1[-1; -2 -3] := U[-1 -3; 1] * sqrt(s)[1; -2]
    @tensor S2[-1; -2 -3] := sqrt(s)[-1; 1] * V[1; -2 -3]
    return S1, S2
end

function Ψ_B(scheme::Loop_TNR, trunc::TensorKit.TruncationScheme)
    ΨA = Ψ_A(scheme)
    ΨB = []

    for i in 1:4
        s1, s2 = SVD12(ΨA[i], trunc)
        push!(ΨB, s1)
        push!(ΨB, s2)
    end

    PR_list, PL_list = find_projectors(ΨB, 100, 1e-12, trunc)

    ΨB_disentangled = []
    for i in 1:8
        @tensor B1[-1; -2 -3] := PL_list[i][-1; 1] * ΨB[i][1; 2 -3] *
                                 PR_list[mod(i, 8) + 1][2; -2]
        push!(ΨB_disentangled, B1)
    end
    return ΨB_disentangled
end

#Entanglement Filtering 

function entanglement_filtering!(scheme::Loop_TNR, maxsteps::Int, minerror::Float64,
                                 trunc::TensorKit.TruncationScheme)
    ΨA = Ψ_A(scheme)
    PR_list, PL_list = find_projectors(ΨA, maxsteps, minerror, trunc)

    TA = copy(scheme.TA)
    TB = copy(scheme.TB)

    @tensor scheme.TA[-1 -2; -3 -4] := TA[1 2; 3 4] * PR_list[4][1; -1] *
                                       PL_list[1][-2; 2] * PR_list[2][3; -3] *
                                       PL_list[3][-4; 4]
    @tensor scheme.TB[-1 -2; -3 -4] := TB[1 2; 3 4] * PL_list[2][-1; 1] *
                                       PR_list[3][2; -2] * PL_list[4][-3; 3] *
                                       PR_list[1][4; -4]

    return scheme
end

#cost functions

function const_C(psiA)
    @tensor tmp[-1 -2; -3 -4] := psiA[1][-1; -3 1 2] * adjoint(psiA[1])[-4 1 2; -2]
    for i in 2:4
        @tensor tmp[-1 -2; -3 -4] := tmp[-1 -2; 1 2] * psiA[i][1; -3 3 4] *
                                     adjoint(psiA[i])[-4 3 4; 2]
    end
    return @tensor tmp[1 2; 1 2]
end

function TNT(pos, psiB)
    @tensor tmp[-1 -2; -3 -4] := psiB[mod(pos, 8) + 1][-2; -1 1] *
                                 adjoint(psiB[mod(pos, 8) + 1])[-3 1; -4]
    tmp = permute(tmp, (2, 1), (4, 3))
    for i in (pos + 1):(pos + 7)
        ΨB = permute(psiB[mod(i, 8) + 1], (1,), (3, 2))
        ΨBdag = permute(adjoint(psiB[mod(i, 8) + 1]), (1, 3), (2,))
        @tensor tmp[-1 -2; -3 -4] := tmp[-1 1; -3 2] * ΨB[1; 3 -2] * ΨBdag[-4 2; 3]
    end
    return @tensor tmp[1 1; 2 2]
end

function WdT(pos, psiA, psiB)
    next_a = mod(ceil(Int, pos / 2), 4) + 1
    next_b = mod(2 * ceil(Int, pos / 2) + 1, 8)

    @tensor tmp[-2 -1; -4 -3] := psiA[next_a][-1; -2 1 2] * adjoint(psiB[next_b])[3 2; -3] *
                                 adjoint(psiB[next_b + 1])[-4 1; 3]
    tmp = permute(tmp, (2, 1), (4, 3))
    for i in next_a:(next_a + 2)
        ΨA = permute(psiA[mod(i, 4) + 1], (1,), (4, 3, 2))
        ΨB1 = permute(psiB[2 * (mod(i, 4) + 1) - 1], (1,), (3, 2))
        ΨB2 = permute(psiB[2 * (mod(i, 4) + 1)], (1,), (3, 2))
        @tensor tmp[-1 -2; -3 -4] := tmp[-1 1; -3 4] * ΨA[1; 2 3 -2] * conj(ΨB1[4; 2 5]) *
                                     conj(ΨB2[5; 3 -4])
    end

    return @tensor tmp[1 1; 2 2]
end

function dWT(pos, psiA, psiB)
    next_a = mod(ceil(Int, pos / 2), 4) + 1
    next_b = mod(2 * ceil(Int, pos / 2) + 1, 8)

    @tensor tmp[-2 -1; -4 -3] := psiB[next_b][-1; 1 2] * psiB[next_b + 1][1; -2 3] *
                                 adjoint(psiA[next_a])[-4 3 2; -3]
    tmp = permute(tmp, (2, 1), (4, 3))
    for i in next_a:(next_a + 2)
        ΨA = permute(psiA[mod(i, 4) + 1], (1,), (4, 3, 2))
        ΨB1 = permute(psiB[2 * (mod(i, 4) + 1) - 1], (1,), (3, 2))
        ΨB2 = permute(psiB[2 * (mod(i, 4) + 1)], (1,), (3, 2))
        @tensor tmp[-1 -2; -3 -4] := tmp[-1 1; -3 4] * ΨB1[1; 3 2] * ΨB2[2; 5 -2] *
                                     adjoint(ΨA)[3 5 -4; 4]
    end

    return @tensor tmp[1 1; 2 2]
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
    @tensor tmp[-1 -2; -3 -4] := psiB[mod(pos, 8) + 1][-2; -1 1] *
                                 adjoint(psiB[mod(pos, 8) + 1])[-3 1; -4]
    tmp = permute(tmp, (2, 1), (4, 3))
    for i in (pos + 1):(pos + 6)
        ΨB = permute(psiB[mod(i, 8) + 1], (1,), (3, 2))
        ΨBdag = permute(adjoint(psiB[mod(i, 8) + 1]), (1, 3), (2,))
        @tensor tmp[-1 -2; -3 -4] := tmp[-1 1; -3 2] * ΨB[1; 3 -2] * ΨBdag[-4 2; 3]
    end
    return tmp
end

function tW(pos, psiA, psiB)
    next_a = mod(ceil(Int, pos / 2), 4) + 1
    next_b = mod(2 * ceil(Int, pos / 2) + 1, 8)

    @tensor tmp[-2 -1; -4 -3] := psiA[next_a][-1; -2 1 2] * adjoint(psiB[next_b])[3 2; -3] *
                                 adjoint(psiB[next_b + 1])[-4 1; 3]
    tmp = permute(tmp, (2, 1), (4, 3))
    for i in next_a:(next_a + 1)
        ΨA = permute(psiA[mod(i, 4) + 1], (1,), (4, 3, 2))
        ΨB1 = permute(psiB[2 * (mod(i, 4) + 1) - 1], (1,), (3, 2))
        ΨB2 = permute(psiB[2 * (mod(i, 4) + 1)], (1,), (3, 2))
        @tensor tmp[-1 -2; -3 -4] := tmp[-1 1; -3 4] * ΨA[1; 2 3 -2] * conj(ΨB1[4; 2 5]) *
                                     conj(ΨB2[5; 3 -4])
    end

    if pos % 2 == 0
        ΨA = permute(psiA[ceil(Int, pos / 2)], (4, 2, 1), (3,))
        ΨB = permute(psiB[pos - 1], (2, 3), (1,))
        tmp = permute(tmp, (3,), (4, 1, 2))
        @tensor W[-1; -2 -3] := tmp[-1; 1 2 3] * conj(ΨB[-2 4; 1]) * ΨA[4 2 3; -3]
        W = permute(W, (2,), (1, 3))
    else
        ΨA = permute(psiA[ceil(Int, pos / 2)], (3, 2, 1), (4,))
        ΨB = permute(psiB[pos + 1], (1, 3), (2,))
        tmp = permute(tmp, (4,), (3, 1, 2))
        @tensor W[-1; -2 -3] := tmp[-1; 1 2 3] * ΨA[4 2 3; -3] * conj(ΨB[-2 4; 1])
    end

    return W
end

function opt_T(N, W, psi)
    function apply_f(x::TensorMap)
        x = permute(x, (1,), (3, 2))
        @tensor b[-1; -2 -3] := N[1 2; -1 -2] * x[2; -3 1]
        b = permute(b, (2,), (1, 3))
        return b
    end

    new_T, info = linsolve(apply_f, W, psi)
    return new_T
end

function loop_opt!(scheme::Loop_TNR, maxsteps_opt::Int, minerror_opt::Float64,
                   trunc::TensorKit.TruncationScheme, verbosity::Int)
    psi_A = Ψ_A(scheme)
    psi_B = Ψ_B(scheme, trunc)

    cost = Inf
    sweep = 0
    while abs(cost) > minerror_opt && sweep < maxsteps_opt
        for i in 1:8
            N = tN(i, psi_B)
            W = tW(i, psi_A, psi_B)
            new_T = opt_T(N, W, psi_B[i])
            psi_B[i] = new_T
        end
        sweep += 1
        cost = cost_func(1, psi_A, psi_B)
        if verbosity > 1
            @infov 3 "Sweep: $sweep, Cost: $cost"
        end
    end
    Ψ5 = permute(psi_B[5], (2,), (1, 3))
    Ψ8 = permute(psi_B[8], (1,), (3, 2))
    Ψ1 = permute(psi_B[1], (1, 2), (3,))
    Ψ4 = permute(psi_B[4], (1,), (3, 2))

    @tensor T1[-1 -2; -3 -4] := Ψ5[-1; 4 1] * Ψ8[-2; 1 2] * Ψ1[2 -4; 3] * Ψ4[-3; 3 4]
    scheme.TA = permute(T1, (1, 2), (4, 3))

    Ψ2 = permute(psi_B[2], (1,), (3, 2))
    Ψ3 = permute(psi_B[3], (2,), (1, 3))
    Ψ6 = permute(psi_B[6], (1,), (3, 2))
    Ψ7 = permute(psi_B[7], (1,), (3, 2))

    @tensor T2[-1 -2; -3 -4] := Ψ2[-1; 4 1] * Ψ3[-2; 1 2] * Ψ6[-4; 2 3] * Ψ7[3; 4 -3]
    scheme.TB = permute(T2, (1, 2), (4, 3))
    return scheme
end

function step!(scheme::Loop_TNR, trunc::TensorKit.TruncationScheme, maxsteps::Int,
               minerror::Float64,
               maxsteps_opt::Int, minerror_opt::Float64, verbosity::Int)
    entanglement_filtering!(scheme, maxsteps, minerror, trunc)
    loop_opt!(scheme, maxsteps_opt, minerror_opt, trunc, verbosity::Int)
    return scheme
end

#2x2 finalise function

function finalize!(scheme::Loop_TNR)
    n = norm(@plansor opt = true scheme.TA[1 2; 3 4] * scheme.TB[3 5; 1 6] *
                                 scheme.TB[7 4; 8 2] * scheme.TA[8 6; 7 5])

    scheme.TA /= n^(1 / 4)
    scheme.TB /= n^(1 / 4)
    return n^(1 / 4)
end

function Base.show(io::IO, scheme::Loop_TNR)
    println(io, "Loop_TNR - Loop Tensor Network Renormalization")
    println(io, "  * TA: $(summary(scheme.TA))")
    println(io, "  * TB: $(summary(scheme.TB))")
    return nothing
end

function run!(scheme::Loop_TNR, trscheme::TensorKit.TruncationScheme, criterion::stopcrit,
              loop_sweeps::Int, loop_error::Float64;
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
            step!(scheme, trscheme, 50, 1e-12, loop_sweeps, loop_error, verbosity)
            push!(data, scheme.finalize!(scheme))
            steps += 1
            crit = criterion(steps, data)
        end

        @infov 1 "Simulation finished\n $(stopping_info(criterion, steps, data))\n Elapsed time: $(t)s\n Iterations: $steps"
        # @infov 1 "Elapsed time: $(t)s"
    end
    return data
end

function run!(scheme::Loop_TNR, trscheme::TensorKit.TruncationScheme;
              finalize_beginning=true, verbosity=1)
    return run!(scheme, trscheme, maxiter(9), 50, 1e-12;
                finalize_beginning=finalize_beginning,
                verbosity=verbosity)
end
