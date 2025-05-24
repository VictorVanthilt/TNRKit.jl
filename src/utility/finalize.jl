const simple_scheme = Union{TRG,ATRG,HOTRG}

# 1x1 unitcell finalize
function finalize!(scheme::simple_scheme)
    n = norm(@tensor scheme.T[1 2; 2 1])
    scheme.T /= n
    return n
end

function finalize!(scheme::BTRG)
    n = norm(@tensor scheme.T[1 2; 4 3] * scheme.S1[4; 2] * scheme.S2[3; 1])
    scheme.T /= n
    return n
end

# 2x2 unitcell finalize
function finalize_two_by_two!(scheme::simple_scheme)
    n = norm(@tensor scheme.T[7 1; 5 4] * scheme.T[4 2; 6 7] * scheme.T[3 6; 2 8] *
                     scheme.T[8 5; 1 3])

    scheme.T /= (n^(1 / 4))
    return n^(1 / 4)
end

function finalize_two_by_two!(scheme::BTRG)
    n′ = @tensor begin
        scheme.T[11 1; 9 8] *
        scheme.S2[8; 2] *
        scheme.T[2 6; 10 11] *
        scheme.S1[3; 6] *
        scheme.T[7 10; 3 12] *
        scheme.S2[4; 7] *
        scheme.T[12 9; 5 4] *
        scheme.S1[5; 1]
    end
    n = norm(n′)
    scheme.T /= (n^(1 / 4))
    return n^(1 / 4)
end

function finalize!(scheme::LoopTNR)
    T1 = permute(scheme.TA, ((1, 2), (4, 3)))
    T2 = permute(scheme.TB, ((1, 2), (4, 3)))
    n = norm(@tensor opt = true T1[1 2; 3 4] * T2[3 5; 1 6] *
                                T2[7 4; 8 2] * T1[8 6; 7 5])

    scheme.TA /= n^(1 / 4)
    scheme.TB /= n^(1 / 4)
    return n^(1 / 4)
end

function finalize!(scheme::CTMHOTRG)
    n = norm(@tensor scheme.T[1][1 2; 2 1])
    scheme.T[1] = scheme.T[1] / n

    scheme.T[1] = permute(scheme.T[1], ((3, 1), (4, 2)))
    scheme.E = PEPSKit.rotate_north(scheme.E, 2)
    return n
end

function finalize!(scheme::ATRG_3D)
    n = norm(@tensor scheme.T[1 1; 2 3 2 3])
    scheme.T /= n
    return n
end

function finalize!(scheme::SLoopTNR)
    tr_norm = trnorm_2x2(scheme.T)
    scheme.T /= tr_norm^0.25
    return tr_norm^0.25
end

# cft data finalize
function finalize_cftdata!(scheme::LoopTNR)
    finalize!(scheme)
    return cft_data(scheme; is_real=true)
end

function finalize_cftdata!(scheme::CTMHOTRG)
    n = norm(@tensor scheme.T[1][1 2; 2 1])
    scheme.T[1] = scheme.T[1] / n

    scheme.T[1] = permute(scheme.T[1], ((3, 1), (4, 2)))
    scheme.E = PEPSKit.rotate_north(scheme.E, 2)
    return cft_data(TRG(scheme.T[1]); is_real=false)
end

function finalize_cft!(scheme::SLoopTNR)
    tr_norm = trnorm_2x2(scheme.T)
    scheme.T /= tr_norm^0.25
    Tflip = flip(scheme.T, (1, 2, 3, 4))
    @tensoropt mat[-1 -2; -3 -4] := scheme.T[1 3; -1 2] * Tflip[1 4; -2 2] *
                                    Tflip[5 3; -3 6] * scheme.T[5 4; -4 6]
    val, vec = eig(mat)
    val = sort(real(val).data; rev=true)
    data = -log.(abs.(val ./ val[1])) / 2 / π
    return data
end

# central charge finalize
function finalize_central_charge!(scheme::TNRScheme)
    n = finalize!(scheme)
    return central_charge(scheme, n)
end
