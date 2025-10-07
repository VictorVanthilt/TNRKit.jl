#Utility functions for QR decomp

function QR_L(L::TensorMap, T::AbstractTensorMap{E, S, M, N}, in_ind, out_ind) where {E, S, M, N}
    permT = (
        (in_ind,),
        (
            reverse(collect(1:(in_ind - 1)))..., collect((M + 1):(M + N))...,
            reverse(collect((in_ind + 1):M))...,
        ),
    )
    permLT = (
        (
            reverse(collect(2:(in_ind + out_ind - 1)))..., 1,
            reverse(collect((in_ind + out_ind + 1):(M + N)))...,
        ), (in_ind + out_ind,),
    )
    LT = transpose(L * transpose(T, permT), permLT)
    _, Rt = leftorth(LT)
    return Rt / norm(Rt, Inf)
end

function QR_R(R::TensorMap, T::AbstractTensorMap{E, S, M, N}, in_ind, out_ind) where {E, S, M, N}
    permT = (
        (
            reverse(collect((M + 1):(M + in_ind - 1)))..., collect(1:M)...,
            reverse(collect((M + in_ind + 1):(M + N)))...,
        ), (M + in_ind,),
    )
    permTR = (
        (in_ind + out_ind - 1,),
        (
            reverse(collect(1:(in_ind + out_ind - 2)))..., M + N,
            reverse(collect((in_ind + out_ind):(M + N - 1)))...,
        ),
    )
    TR = transpose(transpose(T, permT) * R, permTR)
    Lt, _ = rightorth(TR)
    return Lt / norm(Lt, Inf)
end

# Functions to find the left and right projectors

# Function to find the list of left projectors L_list
function find_L(psi::Array, in_inds::Array, out_inds::Array, entanglement_criterion::stopcrit)
    type = eltype(psi[1])
    n = length(psi)
    L_list = []

    for i in 1:n
        L = id(type, codomain(psi[i])[in_inds[i]])
        error = [Inf]
        crit = true
        steps = 1
        while crit
            L_last_time = L
            for j in 0:(n - 1)
                running_pos = mod(i + j - 1, n) + 1
                L = QR_L(
                    L, psi[running_pos], in_inds[running_pos],
                    out_inds[running_pos]
                )
            end
            if space(L) == space(L_last_time)
                push!(error, abs(norm(L - L_last_time)))
            end
            crit = entanglement_criterion(steps, error)
            steps += 1
        end
        push!(L_list, L)
    end

    return L_list
end

# Function to find the list of left projectors L_list
function find_R(psi::Array, in_inds::Array, out_inds::Array, entanglement_criterion::stopcrit)
    type = eltype(psi[1])
    n = length(psi)
    R_list = []

    for i in 1:n
        R = id(type, domain(psi[i])[in_inds[i]])
        error = [Inf]
        crit = true
        steps = 1
        while crit
            R_last_time = R
            for j in 0:(n - 1)
                running_pos = mod(i - j - 1, n) + 1
                R = QR_R(R, psi[running_pos], in_inds[running_pos], out_inds[running_pos])
            end
            if space(R) == space(R_last_time)
                push!(error, abs(norm(R - R_last_time)))
            end
            crit = entanglement_criterion(steps, error)
            steps += 1
        end
        push!(R_list, R)
    end

    return R_list
end

# Function to find the projector P_L and P_R
function P_decomp(R::TensorMap, L::TensorMap, trunc::TensorKit.TruncationScheme)
    U, S, V, _ = tsvd(L * R; trunc = trunc, alg = TensorKit.SVD())
    re_sq = pseudopow(S, -0.5)
    PR = R * V' * re_sq
    PL = re_sq * U' * L
    return PR, PL
end

# Function to find the list of projectors
function find_projectors(psi::Array, in_inds::Array, out_inds::Array, entanglement_criterion::stopcrit, trunc::TensorKit.TruncationScheme)
    PR_list = []
    PL_list = []

    n = length(psi)
    L_list = find_L(psi, in_inds, out_inds, entanglement_criterion)
    R_list = find_R(psi, out_inds, in_inds, entanglement_criterion)
    for i in 1:n
        pr, pl = P_decomp(R_list[mod(i - 2, n) + 1], L_list[i], trunc)
        push!(PR_list, pr)
        push!(PL_list, pl)
    end
    return PR_list, PL_list
end

function MPO_disentangled!(psi::Array, in_inds::Array, out_inds::Array, PR_list::Array, PL_list::Array)
    n = length(psi)
    for i in 1:n
        M = length(codomain(psi[i]))
        N = length(domain(psi[i]))
        in_ind = in_inds[i]
        out_ind = out_inds[i]
        permT = (
            (in_ind,),
            (
                reverse(collect(1:(in_ind - 1)))..., collect((M + 1):(M + N))...,
                reverse(collect((in_ind + 1):M))...,
            ),
        )
        permLT = (
            (
                reverse(collect(2:(in_ind + out_ind - 1)))..., 1,
                reverse(collect((in_ind + out_ind + 1):(M + N)))...,
            ), (in_ind + out_ind,),
        )
        permLTR = (
            Tuple(collect(out_ind:(out_ind + M - 1))),
            (
                collect(reverse(1:(out_ind - 1)))..., M + N,
                collect(reverse((out_ind + M):(M + N - 1)))...,
            ),
        )
        LTR = transpose(
            transpose(PL_list[i] * transpose(psi[i], permT), permLT) *
                PR_list[mod(i, n) + 1], permLTR
        )
        psi[i] = LTR
    end
    return
end

function SVD12(T::AbstractTensorMap{E, S, 1, 3}, trunc::TensorKit.TruncationScheme) where {E, S}
    T_trans = transpose(T, ((2, 1), (3, 4)); copy = true)
    U, s, V, e = tsvd(T_trans; trunc = trunc, alg = TensorKit.SVD())
    @plansor S1[-1; -2 -3] := U[-2 -1; 1] * sqrt(s)[1; -3]
    @plansor S2[-1; -2 -3] := sqrt(s)[-1; 1] * V[1; -2 -3]
    return S1, S2
end

function SVD12(T::AbstractTensorMap{E, S, 2, 2}, trunc::TensorKit.TruncationScheme) where {E, S}
    U, s, V, e = tsvd(T; trunc = trunc)
    return U * sqrt(s), sqrt(s) * V
end
