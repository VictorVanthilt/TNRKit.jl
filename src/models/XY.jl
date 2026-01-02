function algebraic_initialization(m::TensorMap, bond::TensorMap)
    @tensor opt = true T[l u; d r] :=
        m[u; Au Bu] *
        bond[Au; Ad] *
        bond[Bu; Bd] *
        m[Bd; Cu r] *
        m'[l Ad; Du] *
        bond[Du; Dd] *
        bond[Cu; Cd] *
        m'[Dd Cd; d]
    return T
end

function classical_XY_U1_symmetric(beta::Float64, charge_trunc::Int)
    FunU1 = U1Space(map(x -> (x => 1), (-charge_trunc):charge_trunc))

    m = ones(Float64, FunU1 ← FunU1 ⊗ FunU1)

    bond = zeros(Float64, FunU1 ← FunU1)

    for sector in fusiontrees(bond)
        charge = sector[1].uncoupled[1].charge
        bond[sector...] .= besseli(charge, beta)
    end

    return algebraic_initialization(m, bond)
end

function classical_XY_O2_symmetric(beta::Float64, charge_trunc::Int)
    FunU1_0 = CU1Space((0, 0) => 1)
    FunU1_1 = CU1Space(((i, 2) => 1 for i in 1:charge_trunc))
    FunU1 = FunU1_0 ⊕ FunU1_1

    m = zeros(Float64, FunU1 ← FunU1 ⊗ FunU1)

    for (to, from) in fusiontrees(m)
        left, right = from.uncoupled
        if (left == right) && left != CU1Irrep(0, 0) && from.coupled == CU1Irrep(0, 0)
            m[to, from] .= sqrt(2)
        else
            m[to, from] .= 1
        end
    end

    bond = zeros(Float64, FunU1 ← FunU1)

    for sector in fusiontrees(bond)
        charge = sector[1].uncoupled[1].j
        bond[sector...] .= besseli(charge, beta)
    end

    return algebraic_initialization(m, bond)
end
