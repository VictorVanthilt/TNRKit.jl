function T_ele(n1::Int, n2::Int, n3::Int, n4::Int, beta::Float64, trunc::Int)
    if n1 + n2 + n3 + n4 != 0
        return ArgumentError("Charge not conserved")
    end
    list = sort([n2, n2 + n3, n2 + n3 + n4])
    nmax = trunc - list[end]
    nmin = -trunc - list[1]
    T = 0.0
    for n in nmin:nmax
        T += besseli(n, beta) * besseli(n + n2, beta) * besseli(n + n2 + n3, beta) * besseli(n + n2 + n3 + n4, beta)
    end
    return T
end

function XY_init(beta::Float64, charge_trunc::Int)
    V = U1Space(map(x -> (x => 1), -charge_trunc:charge_trunc))
    T = zeros(Float64, V ⊗ V ← V ⊗ V)
    for n1 in -charge_trunc:charge_trunc, n2 in -charge_trunc:charge_trunc, n3 in -charge_trunc:charge_trunc
        n4 = -(n1 + n2 + n3)
        if abs(n4) > charge_trunc
            continue
        else
            T[(n1, n2, n4, n3)] .= T_ele(n1, n2, n3, n4, beta, charge_trunc)
        end
    end
    return T
end
