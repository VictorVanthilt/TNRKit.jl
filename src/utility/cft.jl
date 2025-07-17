function next_τ(τ)
    return (τ - 1) / (τ + 1)
end

function cft_data(scheme::TNRScheme; v=1, unitcell=1, is_real=true)
    # make the indices
    indices = [[i, -i, -(i + unitcell), i + 1] for i in 1:unitcell]
    indices[end][4] = 1

    T = ncon(fill(scheme.T, unitcell), indices)

    outinds = Tuple(collect(1:unitcell))
    ininds = Tuple(collect((unitcell + 1):(2unitcell)))

    T = permute(T, outinds, ininds)
    D, _ = eig(T)

    data = zeros(ComplexF64, dim(space(D, 1)))

    i = 1
    for (_, b) in blocks(D)
        for I in LinearAlgebra.diagind(b)
            data[i] = b[I]
            i += 1
        end
    end

    data = sort(data; by=x -> abs(x), rev=true) # sorting by magnitude
    data = filter(x -> real(x) > 0, data) # filtering out negative real values
    data = filter(x -> abs(x) > 1e-12, data) # filtering out small values

    if is_real
        data = real(data)
    end

    return unitcell * (1 / (2π * v)) * log.(data[1] ./ data)
end

function cft_data(scheme::BTRG; v=1, unitcell=1, is_real=true)
    # make the indices
    indices = [[i, -i, -(i + unitcell), i + 1] for i in 1:unitcell]
    indices[end][4] = 1

    @tensor T_unit[-1 -2; -3 -4] := scheme.T[1 2; -3 -4] * scheme.S1[-2; 2] *
                                    scheme.S2[-1; 1]
    T = ncon(fill(T_unit, unitcell), indices)

    outinds = Tuple(collect(1:unitcell))
    ininds = Tuple(collect((unitcell + 1):(2unitcell)))

    T = permute(T, outinds, ininds)
    D, _ = eig(T)

    data = zeros(ComplexF64, dim(space(D, 1)))

    i = 1
    for (_, b) in blocks(D)
        for I in LinearAlgebra.diagind(b)
            data[i] = b[I]
            i += 1
        end
    end

    data = sort(data; by=x -> abs(x), rev=true) # sorting by magnitude
    data = filter(x -> real(x) > 0, data) # filtering out negative real values
    data = filter(x -> abs(x) > 1e-12, data) # filtering out small values

    if is_real
        data = real(data)
    end

    return unitcell * (1 / (2π * v)) * log.(data[1] ./ data)
end

# Function to obtain the "canonical" normalization constant
function area_term(A, B; is_real=true)
    a_in = domain(A)[1]
    b_in = domain(B)[1]
    x0 = rand(a_in ⊗ b_in)

    function f0(x)
        @planar fx[-1 -2] := A[c -1; 1 m] * x[1 2] * B[m -2; 2 c]
        @planar ffx[-1 -2] := B[c -1; 1 m] * fx[1 2] * A[m -2; 2 c]
        return ffx
    end

    spec0, _, _ = eigsolve(f0, x0, 1, :LR; verbosity=0)

    if is_real
        return real(spec0[1])
    else
        return spec0[1]
    end
end

function MPO_opt(TA::TensorMap, TB::TensorMap, trunc::TensorKit.TruncationScheme,
                 truncentanglement::TensorKit.TruncationScheme)
    pretrunc = truncdim(2 * trunc.dim)
    dl, ur = SVD12(TA, pretrunc)
    dr, ul = SVD12(transpose(TB, (2, 4), (1, 3)), pretrunc)

    transfer_MPO = [transpose(dl, (1,), (3, 2)), ur, transpose(ul, (2,), (3, 1)),
                    transpose(dr, (3,), (2, 1))]

    in_inds = [1, 1, 1, 1]
    out_inds = [1, 2, 2, 1]
    MPO_function(steps, data) = abs(data[end])
    criterion = maxiter(10) & convcrit(1e-12, MPO_function)
    PR_list, PL_list = find_projectors(transfer_MPO, in_inds, out_inds, criterion,
                                       trunc & truncentanglement)

    MPO_disentangled!(transfer_MPO, in_inds, out_inds, PR_list, PL_list)
    return transfer_MPO
end

function reduced_MPO(dl::TensorMap, ur::TensorMap, ul::TensorMap, dr::TensorMap,
                     trunc::TensorKit.TruncationScheme)
    @planar temp[-1 -2; -3 -4] := ur[-1; 1 4] *
                                  ul[4; 3 -2] *
                                  dr[-3; 2 1] * dl[2; -4 3]
    D, U = SVD12(temp, trunc)
    @planar translate[-1 -2; -3 -4] := U[-2; 1 -4] * D[-1 1; -3]
    return translate
end

function MPO_action_1x4(TA::TensorMap, TB::TensorMap, x::TensorMap)
    @tensor TTTTx[-1 -2 -3 -4; -5] := x[1 2 3 4; -5] * TA[41 -1; 1 12] *
                                      TB[12 -2; 2 23] *
                                      TA[23 -3; 3 34] * TB[34 -4; 4 41]
    return TTTTx
end

function MPO_action_1x4_twist(TA::TensorMap, TB::TensorMap, x::TensorMap)
    TTTTx = MPO_action_1x4(TA, TB, x)
    return permute(TTTTx, ((2, 3, 4, 1), (5,)))
end

# Fig.25 of https://arxiv.org/pdf/2311.18785. Firstly appear in Chenfeng Bao's thesis, see http://hdl.handle.net/10012/14674.
function MPO_action_2gates(TA::TensorMap, TB::TensorMap, x::TensorMap)
    @tensor fx[-1 -2 -3 -4; 5] := TB[-1 -2; 1 2] * x[1 2 3 4; 5] * TB[-3 -4; 3 4]
    @tensor ffx[-1 -2 -3 -4; 5] := TA[-3 -4; 2 3] * fx[1 2 3 4; 5] *
                                   TA[-1 -2; 4 1]
    return permute(ffx, (2, 3, 4, 1), (5,))
end

function spec(TA::TensorMap, TB::TensorMap, shape::Array; Nh=25)
    area = shape[1] * shape[2]
    Reτ = shape[1] / shape[2]
    relative_shift = shape[3] / shape[1]

    I = sectortype(TA)
    𝔽 = field(TA)
    if BraidingStyle(I) != Bosonic()
        throw(ArgumentError("Sectors with non-Bosonic charge $I has not been implemented"))
    end

    spec_sector = Dict()
    conformal_data = Dict()

    for charge in values(I)
        if I == Trivial
            V = 𝔽^1
        else
            V = Vect[I](charge => 1)
        end

        if shape ≈ [1, 4, 1]
            x = rand(domain(TA)[1] ⊗ domain(TB)[1] ⊗ domain(TA)[1] ⊗ domain(TB)[1] ← V)
            f = MPO_action_1x4_twist
        elseif shape ≈ [1, 8, 1]
            x = rand(domain(TA)[1] ⊗ domain(TB)[1] ⊗ domain(TA)[1] ⊗ domain(TB)[1] ← V)
            f = MPO_action_1x4
        elseif shape ≈ [sqrt(2), 2 * sqrt(2), 0] ||
               shape ≈ [4/sqrt(10), 2*sqrt(10), 2/sqrt(10)]
            x = rand(domain(TB) ⊗ domain(TB) ← V)
            f = MPO_action_2gates
        end

        if dim(x) == 0
            spec_sector[charge] = [0.0]
        else
            spec, _, _ = eigsolve(a -> f(TA, TB, a), x, Nh, :LM; krylovdim=40, maxiter=100,
                                  tol=1e-12,
                                  verbosity=0)

            spec_sector[charge] = filter(x -> abs(real(x)) ≥ 1e-12, spec)
        end
    end

    norm_const_0 = spec_sector[one(I)][1]
    conformal_data["c"] = 6 / pi / (Reτ - area / 4) * log(norm_const_0)

    for charge in values(I)
        DeltaS = -1 / (2 * pi * shape[1] / shape[2]) *
                 log.(spec_sector[charge] / norm_const_0)
        if !(relative_shift ≈ 0)
            conformal_data[charge] = real.(DeltaS) + imag.(DeltaS) / relative_shift * im
        else
            conformal_data[charge] = DeltaS
        end
    end
    return conformal_data
end

# The function to obtain central charge and conformal spectrum from the fixed-point tensor with G-symmetry. Here the conformal spectrum is obtained by different charge sectors.
# The case with spin is based on https://arxiv.org/pdf/1512.03846 and some private communications with Yingjie Wei and Atsushi Ueda
function cft_data!(scheme::LoopTNR, shape::Array,
                   trunc::TensorKit.TruncationScheme,
                   truncentanglement::TensorKit.TruncationScheme)
    if !(shape in [[1, 8, 1], [4/sqrt(10), 2*sqrt(10), 2/sqrt(10)]])
        throw(ArgumentError("The shape $shape is not correct."))
    end

    norm_const = area_term(scheme.TA, scheme.TB)
    scheme.TA = scheme.TA / norm_const^(1 / 4)
    scheme.TB = scheme.TB / norm_const^(1 / 4)
    @infov 2 "CFT data calculating"

    dl, ur, ul, dr = MPO_opt(scheme.TA, scheme.TB, trunc, truncentanglement)
    T = reduced_MPO(dl, ur, ul, dr, trunc)

    # Calculate conformal data with spin from -4 to 4. Most error is introduced in the second step of the SVD.
    conformal_data = spec(T, T, shape)
    return conformal_data
end

function cft_data!(scheme::LoopTNR, shape::Array)
    if !(shape in [[1, 4, 1], [sqrt(2), 2 * sqrt(2), 0]])
        throw(ArgumentError("The shape $shape is not correct."))
    end

    norm_const = area_term(scheme.TA, scheme.TB)
    scheme.TA = scheme.TA / norm_const^(1 / 4)
    scheme.TB = scheme.TB / norm_const^(1 / 4)
    @infov 2 "CFT data calculating"
    conformal_data = spec(scheme.TA, scheme.TB, shape)
    return conformal_data
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
    @tensor M[-1; -2] := ((scheme.T)[1 -1; 3 2] * scheme.S1[3; -2] *
                          scheme.S2[2; 1]) / n
    _, S, _ = tsvd(M)
    return log(S.data[1]) * 6 / (π)
end
