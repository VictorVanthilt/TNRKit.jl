function next_τ(τ)
    return (τ-1)/(τ+1)
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

function cft_data(scheme::LoopTNR; is_real=true)
    @tensor opt = true T[-1 -2; -3 -4] := scheme.TA[-1 1; 3 2] * scheme.TB[2 6; 4 -3] *
                                          scheme.TB[-2 3; 1 5] * scheme.TA[5 4; 6 -4]

    D, V = eig(T)
    diag = []
    for (i, d) in blocks(D)
        push!(diag, d...)
    end
    diag = filter(x -> abs(x) > 1e-12, diag)
    data = sort!(diag; by=x -> abs(x), rev=true)

    if is_real
        data = real(data)
        return (1 / (2π)) * log.(abs.(data[1] ./ data))
    else
        return (1 / (2π)) * log.(data[1] ./ data)
    end
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
