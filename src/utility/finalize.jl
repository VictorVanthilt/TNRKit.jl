const simple_scheme = Union{TRG}
const turning_scheme = Union{HOTRG,ATRG}

# 1x1 unitcell finalize
function finalize!(scheme::simple_scheme)
    n = norm(@tensor scheme.T[1 2; 2 1])
    scheme.T /= n
    return n
end

function finalize!(scheme::turning_scheme)
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

function finalize_two_by_two!(scheme::turning_scheme) # TODO: update to new convention
    n = norm(@tensor scheme.T[7 1; 5 4] * scheme.T[4 2; 6 7] * scheme.T[3 6; 2 8] *
                     scheme.T[8 5; 1 3])
    scheme.T /= (n^(1 / 4))

    # turn the tensor by 90 degrees
    # scheme.T = permute(scheme.T, ((2, 3), (4, 1)))
    return n^(1 / 4)
end

function finalize_two_by_two!(scheme::BTRG) # TODO: update to new convention
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
    return n ^ (1 / 4)
end
