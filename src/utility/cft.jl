function cft_data(scheme::TNRScheme; v=1, unitcell=1, is_real=true)
    # make the indices
    indices = [[i, -i, i + 1, -(i + unitcell)] for i in 1:unitcell]
    indices[end][3] = 1
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

    data = filter(x -> abs(x) > 1e-12, data)
    data = sort(data; by=x -> abs(x), rev=true)
    if is_real
        data = real(data)
    end
    return unitcell * (1 / (2π * v)) * log.(data[1] ./ data)
end

function cft_data(scheme::BTRG; v=1, unitcell=1, is_real=true)
    # make the indices
    indices = [[i, -i, i + 1, -(i + unitcell)] for i in 1:unitcell]
    indices[end][3] = 1

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
    data = filter(x -> abs(x) > 1e-12, data)
    data = sort(data; by=x -> abs(x), rev=true)
    if is_real
        data = real(data)
    end
    return unitcell * (1 / (2π * v)) * log.(data[1] ./ data)
end

function cft_data(scheme::LoopTNR, isreal=true)
    @tensor opt = true T[-1 -2; -3 -4] := scheme.TA[-1 1; 2 5] * scheme.TB[2 3; -3 6] *
                                          scheme.TB[-2 5; 4 1] * scheme.TA[4 6; -4 3]

    D, V = eig(T)
    diag = []
    for (i, d) in blocks(D)
        push!(diag, d...)
    end
    data = sort!(diag; by=x -> abs(x), rev=true)

    if is_real
        data = real(data)
    end
    return (1 / (2π)) * log.(data[1] ./ data)
end

function central_charge(scheme::TNRScheme, trunc::TensorKit.TruncationScheme,
                        stop::stopcrit)
    data = run!(scheme, trunc, stop; finalize_beginning=true)
    @tensor M[-1; -2] := (scheme.T / data[end])[1 -1; 1 -2]
    _, S, _ = tsvd(M)
    return log(S[1, 1]) * 6 / (π)
end

function central_charge(scheme::BTRG, trunc::TensorKit.TruncationScheme, stop::stopcrit)
    data = run!(scheme, trunc, stop; finalize_beginning=true)
    @tensor M[-1; -2] := (scheme.T / data[end])[1 2; 3 -2] * scheme.S1[-1; 2] *
                         scheme.S2[3; 1]
    _, S, _ = tsvd(M)
    return log(S[1, 1]) * 6 / (π)
end

# default maxiter criterion of 15 iterations
function central_charge(scheme::TNRScheme, trunc::TensorKit.TruncationScheme)
    return central_charge(scheme, trunc, maxiter(15))
end
