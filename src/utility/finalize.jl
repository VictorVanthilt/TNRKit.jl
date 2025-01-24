const simple_scheme = Union{TRG,GILTTNR}
const turning_scheme = Union{HOTRG,ATRG}

# 1x1 unitcell finalize
function finalize!(scheme::simple_scheme)
    n = norm(@tensor scheme.T[1 2; 1 2])
    scheme.T /= n
    return n
end

function finalize!(scheme::turning_scheme)
    n = norm(@tensor scheme.T[1 2; 1 2])
    scheme.T /= n

    # turn the tensor by 90 degrees
    scheme.T = permute(scheme.T, ((2, 3), (4, 1)))
    return n
end

function finalize!(scheme::BTRG)
    n = norm(@tensor scheme.T[1 2; 3 4] * scheme.S1[4; 2] * scheme.S2[3; 1])
    scheme.T /= n
    return n
end

# 2x2 unitcell finalize
function finalize_two_by_two!(scheme::simple_scheme)
    n = norm(@tensor scheme.T[2 5; 1 7] * scheme.T[1 6; 2 8] * scheme.T[3 8; 4 6] *
                     scheme.T[4 7; 3 5])
    scheme.T /= (n^(1 / 4))
    return n
end

function finalize_two_by_two!(scheme::turning_scheme)
    n = norm(@tensor scheme.T[2 5; 1 7] * scheme.T[1 6; 2 8] * scheme.T[3 8; 4 6] *
                     scheme.T[4 7; 3 5])
    scheme.T /= (n^(1 / 4))

    # turn the tensor by 90 degrees
    scheme.T = permute(scheme.T, ((2, 3), (4, 1)))
    return n
end

function finalize_two_by_two!(scheme::BTRG)
    n′ = @tensor begin
        scheme.T[3 7; 1 11] *
        scheme.S2[1; 2] *
        scheme.T[2 9; 3 12] *
        scheme.S1[10; 9] *
        scheme.T[5 12; 6 10] *
        scheme.S2[4; 5] *
        scheme.T[6 11; 4 8] *
        scheme.S1[8; 7]
    end
    n = norm(n′)
    scheme.T /= (n^(1 / 4))
    return n
end
