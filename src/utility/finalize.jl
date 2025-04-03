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

# cft data finalize
function finalize_cftdata!(scheme::TNRScheme)
    finalize!(scheme)
    return cft_data(scheme)
end

# central charge finalize
function finalize_central_charge!(scheme::TNRScheme)
    n = finalize!(scheme)
    return central_charge(scheme, n)
end
