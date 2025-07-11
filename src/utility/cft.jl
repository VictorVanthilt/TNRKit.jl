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
function shape_factor_2x2(A, B; is_real=true)
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

# Fig.25 of https://arxiv.org/pdf/2311.18785. Firstly appear in Chenfeng Bao's thesis, see http://hdl.handle.net/10012/14674.
function spec_2x4(A, B; Nh=10, is_real=true)
    I = sectortype(A)
    𝔽 = field(A)
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
        x = rand(domain(B) ⊗ domain(B) ← V)
        if dim(x) == 0
            spec_sector[charge] = [0.0]
        else
            function f(x)
                @tensor fx[-1 -2 -3 -4; 5] := B[-1 -2; 1 2] * x[1 2 3 4; 5] * B[-3 -4; 3 4]
                @tensor ffx[-1 -2 -3 -4; 5] := A[-3 -4; 2 3] * fx[1 2 3 4; 5] *
                                               A[-1 -2; 4 1]
                return permute(ffx, (2, 3, 4, 1), (5,))
            end
            spec, _, _ = eigsolve(f, x, Nh, :LM; krylovdim=40, maxiter=100, tol=1e-12,
                                  verbosity=0)
            if is_real
                spec_sector[charge] = filter(≥(1e-12), abs.(spec))
            else
                spec_sector[charge] = filter(x -> abs(real(x)) ≥ 1e-12, spec)
            end
        end
    end

    norm_const_0 = spec_sector[one(I)][1]
    conformal_data["c"] = -12 / pi * log(norm_const_0)
    for irr_center in values(I)
        conformal_data[irr_center] = -1 / pi * log.(spec_sector[irr_center] / norm_const_0)
    end
    return conformal_data
end

# The function to obtain central charge and conformal spectrum from the fixed-point tensor with G-symmetry. Here the conformal spectrum is obtained by different charge sectors.
function cft_data!(scheme::LoopTNR; is_real=true)
    norm_const = shape_factor_2x2(scheme.TA, scheme.TB; is_real)
    scheme.TA = scheme.TA / norm_const^(1 / 4)
    scheme.TB = scheme.TB / norm_const^(1 / 4)
    conformal_data = spec_2x4(scheme.TA, scheme.TB; is_real)
    return conformal_data
end

function transfer_MPS(scheme::LoopTNR)
    TA = scheme.TA
    TB = scheme.TB

    if BraidingStyle(sectortype(TA)) isa NoBraiding
        throw(ArgumentError("Transfer MPS is only implemented for sectors with braiding"))
    end

    @planar T1[-1; -2 -3 -4] := TA[1 2; -3 -4] * τ[-2 -1; 1 2]
    @planar T2[-1; -2 -3 -4] := TB[-1 1; -2 2] * τ[-3 2; -4 1]

    return [T1, T2]
end

function planar_opt(scheme::LoopTNR, trunc::TensorKit.TruncationScheme,
                    truncentanglement::TensorKit.TruncationScheme)
    TA = scheme.TA
    TB = scheme.TB

    pretrunc = truncdim(2 * trunc.dim)
    # Perform SVD on the tensors
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
    return vcat(transfer_MPO, transfer_MPO, transfer_MPO, transfer_MPO)
end

function circular_opt(scheme::LoopTNR, trunc::TensorKit.TruncationScheme,
             truncentanglement::TensorKit.TruncationScheme)
    ΨA = Ψ_A(scheme)
    NA = length(ΨA)
    ΨB = []
    for i in 1:NA
        s1, s2 = SVD12(ΨA[i], truncdim(trunc.dim * 2))
        push!(ΨB, s1)
        push!(ΨB, s2)
    end

    ΨB_function(steps, data) = abs(data[end])
    criterion = maxiter(10) & convcrit(1e-12, ΨB_function)
    in_inds = ones(Int, 2*NA)
    out_inds = 2*ones(Int, 2*NA)
    PR_list, PL_list = find_projectors(ΨB, in_inds, out_inds, criterion,
                                       trunc & truncentanglement)
    @planar dl[-1; -2 -3] := ΨB[1][-3; -1 1] *  PR_list[2][1; -2]
    ur = PL_list[2] * ΨB[2]
    @planar ul[-1; -2 -3] := ΨB[3][-1; -2 1] * PR_list[4][1; -3]
    dr = PL_list[4] * ΨB[4]
    return [dl, ur, ul, dr]
end

function transfer_MPO_opt(scheme::LoopTNR, loop_criterion::stopcrit,
                          trunc::TensorKit.TruncationScheme,
                          truncentanglement::TensorKit.TruncationScheme,
                          verbosity::Int)
    psiA = transfer_MPS(scheme)
    # psiB = loop_opt(psiA, loop_criterion, trunc, truncentanglement, verbosity)
    psiB = Ψ_B(psiA, trunc, truncentanglement)

    @planar T1[-1; -2 -3] := psiB[1][1; 2 -2] * τ[2 -1; 1 -3]
    psiB[1] = T1

    @planar T4[-1; -2 -3] := psiB[4][-1; 1 2] * τ[1 2; -2 -3]
    psiB[4] = T4

    return psiB
end

function reduced_MPO(transfer_MPO::Array, trunc::TensorKit.TruncationScheme)
    @planar temp[-1 -2; -3 -4] := transfer_MPO[2][-1; 1 4] *
                                  transfer_MPO[3][4; 3 -2] *
                                  transfer_MPO[4][-3; 2 1] * transfer_MPO[1][2; -4 3]
    D, U = SVD12(temp, trunc)
    @planar translate[-1 -2; -3 -4] := U[-2; 1 -4] * D[-1 1; -3]
    return translate
end

function spec_1x8(T; Nh=10)
    I = sectortype(T)
    𝔽 = field(T)
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
        W = domain(T)[1]
        x = rand(W ⊗ W ⊗ W ⊗ W ← V)
        if dim(x) == 0
            spec_sector[charge] = [0.0]
        else
            function f(x)
                @tensor TTTTx[-1 -2 -3 -4; -5] := x[1 2 3 4; -5] * T[41 -1; 1 12] *
                                                  T[12 -2; 2 23] *
                                                  T[23 -3; 3 34] * T[34 -4; 4 41]
                return TTTTx
            end
            spec, _, _ = eigsolve(f, x, Nh, :LM; krylovdim=40, maxiter=100, tol=1e-12,
                                  verbosity=0)

            spec_sector[charge] = filter(x -> abs(real(x)) ≥ 1e-12, spec)
        end
    end

    norm_const_0 = spec_sector[one(I)][1]
    conformal_data["c"] = - 16 / 5 / pi * log(norm_const_0)
    for irr_center in values(I)
        conformal_data[irr_center] = -4 / pi * log.(spec_sector[irr_center] / norm_const_0)
    end
    return conformal_data
end

# Based on https://arxiv.org/pdf/1512.03846 and some private communications with Yingjie Wei and Atsuchi Ueda
function cft_data_spin!(scheme::LoopTNR, loop_criterion::stopcrit,
                        trunc::TensorKit.TruncationScheme,
                        truncentanglement::TensorKit.TruncationScheme,
                        verbosity::Int)
    norm_const = shape_factor_2x2(scheme.TA, scheme.TB)
    scheme.TA = scheme.TA / norm_const^(1 / 4)
    scheme.TB = scheme.TB / norm_const^(1 / 4)
    @infov 2 "CFT data calculating"
    # transfer_MPO = transfer_MPO_opt(scheme, loop_criterion, trunc, truncentanglement,
    #                                 verbosity, false)
    transfer_MPO = planar_opt(scheme, trunc, truncentanglement)
    T = reduced_MPO(transfer_MPO, trunc)
    conformal_data = spec_1x8(T)
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
