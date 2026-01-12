function next_τ(τ)
    return (τ - 1) / (τ + 1)
end

function cft_data(scheme::TNRScheme; v = 1, unitcell = 1, is_real = true)
    # make the indices
    indices = [[i, -i, -(i + unitcell), i + 1] for i in 1:unitcell]
    indices[end][4] = 1

    T = ncon(fill(scheme.T, unitcell), indices)

    outinds = Tuple(collect(1:unitcell))
    ininds = Tuple(collect((unitcell + 1):(2unitcell)))

    T = permute(T, (outinds, ininds))
    D, _ = eig(T)

    data = zeros(ComplexF64, dim(space(D, 1)))

    i = 1
    for (_, b) in blocks(D)
        for I in LinearAlgebra.diagind(b)
            data[i] = b[I]
            i += 1
        end
    end

    data = sort(data; by = x -> abs(x), rev = true) # sorting by magnitude
    data = filter(x -> real(x) > 0, data) # filtering out negative real values
    data = filter(x -> abs(x) > 1.0e-12, data) # filtering out small values

    if is_real
        data = real(data)
    end

    return unitcell * (1 / (2π * v)) * log.(data[1] ./ data)
end

function cft_data(scheme::BTRG; v = 1, unitcell = 1, is_real = true)
    # make the indices
    indices = [[i, -i, -(i + unitcell), i + 1] for i in 1:unitcell]
    indices[end][4] = 1

    @tensor T_unit[-1 -2; -3 -4] := scheme.T[1 2; -3 -4] * scheme.S1[-2; 2] *
        scheme.S2[-1; 1]
    T = ncon(fill(T_unit, unitcell), indices)

    outinds = Tuple(collect(1:unitcell))
    ininds = Tuple(collect((unitcell + 1):(2unitcell)))

    T = permute(T, (outinds, ininds))
    D, _ = eig(T)

    data = zeros(ComplexF64, dim(space(D, 1)))

    i = 1
    for (_, b) in blocks(D)
        for I in LinearAlgebra.diagind(b)
            data[i] = b[I]
            i += 1
        end
    end

    data = sort(data; by = x -> abs(x), rev = true) # sorting by magnitude
    data = filter(x -> real(x) > 0, data) # filtering out negative real values
    data = filter(x -> abs(x) > 1.0e-12, data) # filtering out small values

    if is_real
        data = real(data)
    end

    return unitcell * (1 / (2π * v)) * log.(data[1] ./ data)
end

"""
The "canonical" normalization constant for loop-TNR tensors,
which is the eigenvalue with largest norm of the 2 x 2 transfer matrix.
If the system is described by a CFT, this constant is
Λ₀ = exp(-π/6 * ℑ(τ) * c + fA)
where A is the area of the unit block, τ is the modular parameter and c is the central charge.

On the square lattice, we have τ = i and A = 4. On the Kagome lattice, we have τ = exp(2πi/3) and A = 3√3/2.
Here we have assumed each tensor represents a partition function on a diamond shape with angle π/3 and 2π/3 and each edge has lenth 1.
"""
function area_term(A::TensorMap{E, S, 2, 2}, B::TensorMap{E, S, 2, 2}; is_real = true) where {E, S}
    a_in = domain(A)[1]
    b_in = domain(B)[1]
    x0 = ones(a_in ⊗ b_in)

    function f0(x)
        @plansor fx[-1 -2] := A[c -1; 1 m] * x[1 2] * B[m -2; 2 c]
        @plansor ffx[-1 -2] := B[c -1; 1 m] * fx[1 2] * A[m -2; 2 c]
        return ffx
    end

    spec0, _, info = eigsolve(f0, x0, 1, :LM; verbosity = 0)
    if info.converged == 0
        @warn "The area term eigensolver did not converge."
    end
    if is_real                  # If the central charge is real and the ground state is of spin-0
        return norm(spec0[1])
    else
        return spec0[1]
    end
end

# The fixed-point tensor network represents the partition function of the original system on a hexagon:
#         1       2
#           ↘   ↙
#             B
#           ↙   ↘
#         ↙       ↘
# --←-- C ----←---- A --←--
#     ↙               ↘
#   2                   1
#
# =
#
# Z(
#                / \
#           ↙           ⇘
#      /                     \
#      \          B          /
#     |     \           /     |
#     ⇊          \ /          ⇊
#     |     C     |     A     |
#      \          |          /
#           ⇘     |     ↙
#                \ /
# )
function area_term(A::TensorMap{E, S, 2, 2}, B::TensorMap{E, S, 2, 2}, C::TensorMap{E, S, 2, 2}; is_real = true) where {E, S}
    x0 = ones(domain(B))

    function f0(x)
        fx = B * x
        @plansor ffx[-1 -2] := fx[1 2] * A[mid -2; 2 cir] * C[cir -1; 1 mid]
        return permute(ffx, ((2, 1), ()))
    end

    spec0, _, info = eigsolve(f0, x0, 1, :LM; verbosity = 0)
    if info.converged == 0
        @warn "The area term eigensolver did not converge."
    end
    if is_real
        return abs(spec0[1])
    else
        return spec0[1]
    end
end

function MPO_opt(
        TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}, trunc::TensorKit.TruncationScheme,
        truncentanglement::TensorKit.TruncationScheme
    ) where {E, S}
    pretrunc = truncdim(2 * trunc.dim)
    dl, ur = SVD12(TA, pretrunc)
    dr, ul = SVD12(transpose(TB, ((2, 4), (1, 3))), pretrunc)

    transfer_MPO = [
        transpose(dl, ((1,), (3, 2))), ur, transpose(ul, ((2,), (3, 1))),
        transpose(dr, ((3,), (2, 1))),
    ]

    in_inds = [1, 1, 1, 1]
    out_inds = [1, 2, 2, 1]
    MPO_function(steps, data) = abs(data[end])
    criterion = maxiter(10) & convcrit(1.0e-12, MPO_function)
    PR_list, PL_list = find_projectors(
        transfer_MPO, in_inds, out_inds, criterion,
        trunc & truncentanglement
    )

    MPO_disentangled!(transfer_MPO, in_inds, out_inds, PR_list, PL_list)
    return transfer_MPO
end

function reduced_MPO(
        dl::TensorMap{E, S, 1, 2}, ur::TensorMap{E, S, 1, 2}, ul::TensorMap{E, S, 1, 2}, dr::TensorMap{E, S, 1, 2},
        trunc::TensorKit.TruncationScheme
    ) where {E, S}
    @plansor temp[-1 -2; -3 -4] := ur[-1; 1 4] *
        ul[4; 3 -2] *
        dr[-3; 2 1] * dl[2; -4 3]
    D, U = SVD12(temp, trunc)
    @plansor translate[-1 -2; -3 -4] := U[-2; 1 -4] * D[-1 1; -3]
    return translate
end

function MPO_action_1x4(TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}, x::TensorMap{E, S, 4, 1}) where {E, S}
    @tensor contractcheck = true TTTTx[-1 -2 -3 -4; -5] := x[1 2 3 4; -5] * TA[41 -1; 1 12] *
        TB[12 -2; 2 23] *
        TA[23 -3; 3 34] * TB[34 -4; 4 41]
    return TTTTx
end

function MPO_action_1x4_twist(TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}, x::TensorMap{E, S, 4, 1}) where {E, S}
    TTTTx = MPO_action_1x4(TA, TB, x)
    return permute(TTTTx, ((2, 3, 4, 1), (5,)))
end

# Fig.25 of https://arxiv.org/pdf/2311.18785. Firstly appear in Chenfeng Bao's thesis, see http://hdl.handle.net/10012/14674.
function MPO_action_2gates(TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}, x::TensorMap{E, S, 4, 1}) where {E, S}
    @tensor fx[-1 -2 -3 -4; 5] := TB[-1 -2; 1 2] * x[1 2 3 4; 5] * TB[-3 -4; 3 4]
    @tensor ffx[-1 -2 -3 -4; 5] := TA[-3 -4; 2 3] * fx[1 2 3 4; 5] *
        TA[-1 -2; 4 1]
    return permute(ffx, ((2, 3, 4, 1), (5,)))
end
#       1       2         3       4
#         ↘   ↙             ↘   ↙
#           B                 B
#         ↙   ↘             ↙   ↘
# --←-- C --←-- A ---←--- C --←-- A --←--
#     ↙           ↘     ↙           ↘
#   4               1 2               3

function MPO_action_2triangles(TA::TensorMap{E, S, 2, 2}, TB::TensorMap{E, S, 2, 2}, TC::TensorMap{E, S, 2, 2}, x::TensorMap{E, S, 4, 1}) where {E, S}
    @tensor fx[-1 -2 -3 -4; -5] := x[1 2 3 4; -5] * TB[-1 -2; 1 2] * TB[-3 -4; 3 4]
    return MPO_action_1x4_twist(TC, TA, fx)
end

# Assign the corresponding Hilbert space and action functions
function _action_assignmentor_no_approximation(scheme::LinearLoopScheme, shape::Array)
    return if shape ≈ [1, 4, 1]
        return domain(scheme.TA)[1] ⊗ domain(scheme.TB)[1] ⊗ domain(scheme.TA)[1] ⊗ domain(scheme.TB)[1],
            x -> MPO_action_1x4_twist(scheme.TA, scheme.TB, x)
    elseif shape ≈ [sqrt(2), 2 * sqrt(2), 0]
        return domain(scheme.TB) ⊗ domain(scheme.TB), x -> MPO_action_2gates(scheme.TA, scheme.TB, x)
    elseif shape ≈ [3 / 2, 2 * sqrt(3), sqrt(3) / 2]
        return domain(scheme.TB) ⊗ domain(scheme.TB), x -> MPO_action_2triangles(scheme.TA, scheme.TB, scheme.TC, x)
    end
end
function _action_assignmentor_approximation(scheme::LinearLoopScheme, shape::Array, trunc::TensorKit.TruncationScheme, truncentanglement::TensorKit.TruncationScheme)
    return if shape ≈ [1, 8, 1] || shape ≈ [4 / sqrt(10), 2 * sqrt(10), 2 / sqrt(10)]
        dl, ur, ul, dr = MPO_opt(scheme.TA, scheme.TB, trunc, truncentanglement)
        T = reduced_MPO(dl, ur, ul, dr, trunc)
        if shape ≈ [1, 8, 1]
            return domain(T)[1] ⊗ domain(T)[1] ⊗ domain(T)[1] ⊗ domain(T)[1], x -> MPO_action_1x4(T, T, x)
        elseif shape ≈ [4 / sqrt(10), 2 * sqrt(10), 2 / sqrt(10)]
            return domain(T) ⊗ domain(T), x -> MPO_action_2gates(T, T, x)
        end
    end
end

# area0: the area in the unitcell, used in the function area_term.
function cft_data_solver(xspace::TensorSpace, f::Function, shape::Array; Nh = 25, area0 = 4.0, Imτ0 = 1.0)
    I = sectortype(xspace)
    area = shape[1] * shape[2]
    Imτ = shape[1] / shape[2]
    relative_shift = shape[3] / shape[1]

    if BraidingStyle(I) != Bosonic()
        throw(ArgumentError("Sectors with non-Bosonic charge $I has not been implemented"))
    end

    spec_sector = Dict(
        map(sectors(fuse(xspace))) do charge
            V = (I == Trivial) ? 𝔽^1 : Vect[I](charge => 1)
            x = ones(xspace ← V) # Initial guess of the eigenvector
            if dim(x) == 0
                return charge => [0.0]
            else
                spec, _, info = eigsolve(
                    f, x, Nh, :LM; krylovdim = 40, maxiter = 100,
                    tol = 1.0e-12,
                    verbosity = 0
                )
                if info.converged == 0
                    @warn "The spectrum eigensolver in sector $charge did not converge."
                end
                return charge => filter(x -> abs(real(x)) ≥ 1.0e-12, spec)
            end
        end
    )

    norm_const_0 = spec_sector[one(I)][1]
    central_charge = 6 / pi / (Imτ - area / area0 * Imτ0) * log(norm_const_0) # calculate central charge

    conformal_dims = Dict(                                                          # transform eigenvalue data to conformal dimension data
        map(sectors(fuse(xspace))) do charge
            DeltaS = -1 / (2 * pi * Imτ) * log.(spec_sector[charge] / norm_const_0)
            if !(relative_shift ≈ 0)
                return charge => real.(DeltaS) + imag.(DeltaS) / relative_shift * im
                @info "The shape $shape with horizontal displacement $(shape[3]) can only resolve conformal spins up to $(shape[2] / shape[3])"
            else
                @warn "The shape $shape with horizontal displacement $(shape[3])≈0 cannot resolve conformal spins."
                return charge => DeltaS
            end
        end
    )

    return central_charge, conformal_dims       # output is a two-elements tuple
end


# The function to obtain central charge and conformal spectrum from the fixed-point tensor with G-symmetry. Here the conformal spectrum is obtained by different charge sectors.
# The case with spin is based on https://arxiv.org/pdf/1512.03846 and some private communications with Yingjie Wei and Atsushi Ueda
function cft_data(
        scheme::LoopTNR, shape::Array,
        trunc::TensorKit.TruncationScheme;
        truncentanglement = truncbelow(1.0e-13)
    )
    if !(shape in [[1, 8, 1], [4 / sqrt(10), 2 * sqrt(10), 2 / sqrt(10)]])
        throw(ArgumentError("The shape $shape is not correct."))
    end

    norm_const = area_term(scheme.TA, scheme.TB)^(1 / 4) # calculate the normalization constant
    @infov 2 "CFT data calculating"

    # Calculate conformal data with spin from -4 to 4. Most error is introduced in the second step of the SVD.
    V, f = _action_assignmentor_approximation(LoopTNR(scheme.TA / norm_const, scheme.TB / norm_const), shape, trunc, truncentanglement)
    return cft_data_solver(V, f, shape)
end

function cft_data(scheme::LoopTNR, shape::Array)
    if !(shape in [[1, 4, 1], [sqrt(2), 2 * sqrt(2), 0]])
        throw(ArgumentError("The shape $shape is not correct."))
    end

    norm_const = area_term(scheme.TA, scheme.TB)^(1 / 4)
    @infov 2 "CFT data calculating"
    V, f = _action_assignmentor_no_approximation(LoopTNR(scheme.TA / norm_const, scheme.TB / norm_const), shape)
    return cft_data_solver(V, f, shape)
end

function cft_data(scheme::KagomeLoopTNR, shape::Array)
    if !(shape in [[3 / 2, 2 * sqrt(3), sqrt(3) / 2]])
        throw(ArgumentError("The shape $shape is not correct."))
    end
    norm_const = area_term(scheme.TA, scheme.TB, scheme.TC)^(1 / 3)
    V, f = _action_assignmentor_no_approximation(KagomeLoopTNR(scheme.TA / norm_const, scheme.TB / norm_const, scheme.TC / norm_const), shape)
    return cft_data_solver(V, f, shape; area0 = 3 * sqrt(3) / 2, Imτ0 = sqrt(3) / 2)
end

"""
    central_charge(scheme::TNRScheme, n::Number)

Get the central charge given the current state of a `TNRScheme` and the previous normalization factor `n`
"""
function central_charge(scheme::TNRScheme, n::Number)
    @tensor M[-1; -2] := (scheme.T / n)[1 -1; -2 1]
    _, S, _ = tsvd(M)
    return log(S.data[1]) * 6 / (π)
end

function central_charge(scheme::BTRG, n::Number)
    @tensor M[-1; -2] := (
        (scheme.T)[1 -1; 3 2] * scheme.S1[3; -2] *
            scheme.S2[2; 1]
    ) / n
    _, S, _ = tsvd(M)
    return log(S.data[1]) * 6 / (π)
end
