"""
$(SIGNATURES)

returns the inverse critical temperature for the classical q-state Potts model on a 2D square lattice.

See also: [`classical_potts`](@ref), [`classical_potts_symmetric`](@ref).
"""
potts_βc(q) = log(1.0 + sqrt(q))

"""
$(SIGNATURES)

Constructs the partition function tensor for the classical Potts model with `q` states
and a given inverse temperature `β`.

### Examples
```julia
    classical_potts(3) # Default inverse temperature is `potts_βc(3)`
    classical_potts(3, 0.5) # Custom inverse temperature.
```

See also: [`classical_potts_symmetric`](@ref), [`potts_βc`](@ref).
"""
function classical_potts(q::Int, β::Float64)
    V = ℂ^q
    A_potts = TensorMap(zeros, V ⊗ V ← V ⊗ V)

    for i in 1:q
        for j in 1:q
            for k in 1:q
                for l in 1:q
                    E = -(Int(i == j) + Int(j == l) + Int(l == k) + Int(k == i))
                    A_potts[i, j, k, l] = exp(-β * E)
                end
            end
        end
    end
    return A_potts
end
classical_potts(q::Int) = classical_potts(q, potts_βc(q))

function weyl_heisenberg_matrices(Q::Int, elt = ComplexF64)
    U = zeros(elt, Q, Q) # clock matrix
    V = zeros(elt, Q, Q) # shift matrix
    W = zeros(elt, Q, Q) # DFT
    ω = cis(2 * pi / Q)

    for row in 1:Q
        U[row, row] = ω^(row - 1)
        V[row, mod1(row - 1, Q)] = one(elt)
        for col in 1:Q
            W[row, col] = ω^((row - 1) * (col - 1))
        end
    end
    return U, V, W / sqrt(Q)
end

"""
$(SIGNATURES)

Constructs the partition function tensor for a symmetric Potts model with `q` states
and a given inverse temperature `β`.

This tensor has explicit ℤq symmetry on each of its spaces.

### Examples
```julia
    classical_potts_symmetric(3) # Default inverse temperature is `potts_βc(3)`
    classical_potts_symmetric(3, 0.5) # Custom inverse temperature.
```

See also: [`classical_potts`](@ref), [`potts_βc`](@ref).
"""
function classical_potts_symmetric(q::Int64, β::Float64)
    V = ℂ^q
    A_potts = TensorMap(zeros, Float64, V ⊗ V ← V ⊗ V)
    for i in 1:q
        for j in 1:q
            for k in 1:q
                for l in 1:q
                    E = -(Int(i == j) + Int(j == l) + Int(k == l) + Int(k == i))
                    A_potts[i, j, k, l] = exp(-β * E)
                end
            end
        end
    end
    Vp = Vect[ZNIrrep{q}](sector => 1 for sector in 0:(q - 1))
    _, _, W = weyl_heisenberg_matrices(q)
    P = TensorMap(W, ℂ^q ← ℂ^q)
    A_potts = TensorMap(
        reshape(((P' ⊗ P') * A_potts * (P ⊗ P)).data, (q, q, q, q)),
        Vp ⊗ Vp ← Vp ⊗ Vp
    )
    return A_potts
end
classical_potts_symmetric(q::Int) = classical_potts_symmetric(q, potts_βc(q))


"""
$(SIGNATURES)

Constructs the partition function tensor for a Potts model with `q` states
and a given inverse temperature `β` with impurities in sectors `k1` and `k2`.

### Examples
```julia
    classical_potts_impurity(3) # Default inverse temperature is `potts_βc(3)`
    classical_potts_impurity(3, 1, 2) # Custom inverse temperature and impurity sectors.
    classical_potts_impurity(3, 0.5, 2, 3) # Custom inverse temperature and impurity sectors.
```

See also: [`classical_potts`](@ref), [`potts_βc`](@ref).
"""

function classical_potts_impurity(q::Int64, β::Float64, k1::Int64 = 1, k2::Int64 = 1)
    bond_tensor = zeros(ComplexF64, q, q)
    for i in 0:(q - 1)
        bond_tensor[i + 1, i + 1] = sqrt(exp(β) - 1 + q * (i == 0))
    end
    Vp = ℂ^q
    bond_tensor = TensorMap(bond_tensor, Vp ← Vp)

    core_tensor = zeros(ComplexF64, q, q, q, q)
    for (i, j, k, l) in Iterators.product(0:(q - 1), 0:(q - 1), 0:(q - 1), 0:(q - 1))
        core_tensor[i + 1, j + 1, k + 1, l + 1] =
            mod(i + j - k - l + k1 - k2, q) == 0 ? 1 : 0
    end
    core_tensor = TensorMap(core_tensor, Vp ⊗ Vp ← Vp ⊗ Vp)

    @tensor T[-1 -2; -3 -4] :=
        core_tensor[1 2; 3 4] *
        bond_tensor[-1; 1] *
        bond_tensor[-2; 2] *
        bond_tensor[3; -3] *
        bond_tensor[4; -4] * (1 / q)


    return T
end

classical_potts_impurity(q::Int64, k1::Int64, k2::Int64) = classical_potts_impurity(q, potts_βc(q), k1, k2)
classical_potts_impurity(q::Int64) = classical_potts_impurity(q, potts_βc(q), 1, 1)
