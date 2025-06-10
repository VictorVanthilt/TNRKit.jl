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
function shape_factor_2x2(A::AbstractTensorMap, B::AbstractTensorMap; is_real=true)
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
function spec_2x4(A::AbstractTensorMap, B::AbstractTensorMap; Nh=10, is_real=true)
    I = sectortype(A)
    if BraidingStyle(I) != Bosonic()
        throw(ArgumentError("Sectors with non-Bosonic charge $I has not been implemented"))
    end

    spec_sector = Dict()
    conformal_data = Dict()

    for charge in values(I)
        V = Vect[I](charge=>1)
        x = rand(domain(B)⊗domain(B)←V)
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
                spec_sector[charge] = filter(x->abs(x)≥1e-12, real.(spec))
            else
                spec_sector[charge] = filter(x->abs(real(x))≥1e-12, spec)
            end
        end
    end

    norm_const_0 = spec_sector[one(I)][1]
    conformal_data["c"] = -12/pi*log(norm_const_0)
    for charge in values(I)
        conformal_data[charge] = - 1/pi * log.(spec_sector[charge]/norm_const_0)
    end
    return conformal_data
end

# The function to obtain central charge and conformal spectrum from the fixed-point tensor with G-symmetry. Here the conformal spectrum is obtained by different charge sectors.
function cft_data!(scheme::LoopTNR; is_real=true)
    norm_const = shape_factor_2x2(scheme.TA, scheme.TB; is_real)
    scheme.TA = scheme.TA/norm_const^(1/4)
    scheme.TB = scheme.TB/norm_const^(1/4)
    conformal_data = spec_2x4(scheme.TA, scheme.TB; is_real)
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
