const ising_βc = BigFloat(log(BigFloat(1.0) + sqrt(BigFloat(2.0))) / BigFloat(2.0))
const ising_cft_exact = [
    1 / 8, 1, 9 / 8, 9 / 8, 2, 2, 2, 2, 17 / 8, 17 / 8, 17 / 8, 3, 3,
    3, 3, 3,
    25 / 8, 25 / 8, 25 / 8, 25 / 8, 25 / 8, 25 / 8,
]
const ising_βc_3D = 1.0 / 4.51152469

"""
$(SIGNATURES)

Constructs the partition function tensor for a 2D square lattice
for the classical Ising model with a given inverse temperature `β` and external magnetic field `h`.

### Examples
```julia
    classical_ising() # Default inverse temperature is `ising_βc`
    classical_ising(0.5; h = 1.0) # Custom inverse temperature and magnetic field.
```
!!! info
    When calculating the free energy with `free_energy()`, set the `initial_size` keyword argument to `2.0`.
    The initial lattice holds 2 spins.

See also: [`classical_ising_symmetric`](@ref), [`classical_ising_symmetric_3D`](@ref), [`classical_ising_3D`](@ref).
"""
function classical_ising(β::Number; h = 0)
    function σ(i::Int64)
        return 2i - 3
    end

    T_array = Float64[
        exp(
                β * (σ(i)σ(j) + σ(j)σ(l) + σ(l)σ(k) + σ(k)σ(i)) +
                h / 2 * β * (σ(i) + σ(j) + σ(k) + σ(l))
            )
            for i in 1:2, j in 1:2, k in 1:2, l in 1:2
    ]

    T = TensorMap(T_array, ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2)

    return T
end
classical_ising() = classical_ising(ising_βc)

function Ising_magnetisation(β::Number; h = 0)
    init = zeros(ComplexF64, 2, 2, 2, 2)
    for (i, j, k, l) in Iterators.product([1:2 for _ in 1:4]...)
        if mod(i + j + k + l, 2) == 0
            init[i, j, k, l] = sinh(h * β)
        else
            init[i, j, k, l] = cosh(h * β)
        end
    end
    init = TensorMap(init, ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2)

    bond_tensor = zeros(ComplexF64, 2, 2)
    bond_tensor[1, 1] = cosh(β)
    bond_tensor[2, 2] = sinh(β)
    bond_tensor = TensorMap(bond_tensor, ℂ^2 ← ℂ^2)

    @tensor T[-1 -2; -3 -4] := init[1 2; 3 4] * bond_tensor[-1; 1] * bond_tensor[-2; 2] *
        bond_tensor[-3; 3] * bond_tensor[-4; 4]
    return T
end
Ising_magnetisation() = Ising_magnetisation(ising_βc)

"""
$(SIGNATURES)

Constructs the partition function tensor for a symmetric 2D square lattice
for the classical Ising model with a given inverse temperature `β`.

This tensor has explicit ℤ₂ symmetry on each of it spaces.

### Examples
```julia
    classical_ising_symmetric() # Default inverse temperature is `ising_βc`
    classical_ising_symmetric(0.5) # Custom inverse temperature.
```

See also: [`classical_ising`](@ref), [`classical_ising_symmetric_3D`](@ref), [`classical_ising_3D`](@ref).
"""
function classical_ising_symmetric(β)
    x = cosh(β)
    y = sinh(β)

    S = ℤ₂Space(0 => 1, 1 => 1)
    T = zeros(Float64, S ⊗ S ← S ⊗ S)
    block(T, Irrep[ℤ₂](0)) .= [2x^2 2x * y; 2x * y 2y^2]
    block(T, Irrep[ℤ₂](1)) .= [2x * y 2x * y; 2x * y 2x * y]

    return T
end
classical_ising_symmetric() = classical_ising_symmetric(ising_βc)

const f_onsager::BigFloat = -2.10965114460820745966777928351108478082549327543540531781696107967700291143188081390114126499095041781

"""
$(SIGNATURES)

Constructs the partition function tensor for a symmetric 3D cubic lattice
for the classical Ising model with a given inverse temperature `β`.

This tensor has explicit ℤ₂ symmetry on each of its spaces.

### Examples
```julia
    classical_ising_symmetric_3D() # Default inverse temperature is `ising_βc_3D`
    classical_ising_symmetric_3D(0.5) # Custom inverse temperature.
```

See also:  [`classical_ising_3D`](@ref), [`classical_ising`](@ref), [`classical_ising_symmetric`](@ref).
"""
function classical_ising_symmetric_3D(β)
    x = cosh(β)
    y = sinh(β)
    W = [sqrt(x) sqrt(y); sqrt(x) -sqrt(y)]
    T_array = zeros(Float64, 2, 2, 2, 2, 2, 2)
    for (i, j, k, l, m, n) in Iterators.product([1:2 for _ in 1:6]...)
        for a in 1:2
            # Outer product of W[a, :] with itself 6 times
            T_array[i, j, k, l, m, n] += W[a, i] * W[a, j] * W[a, k] * W[a, l] * W[a, m] *
                W[a, n]
        end
    end
    S = ℤ₂Space(0 => 1, 1 => 1)
    T = TensorMap(T_array, S ⊗ S ⊗ S ← S ⊗ S ⊗ S)

    return permute(T, ((1, 4), (5, 6, 2, 3)))
end
classical_ising_symmetric_3D() = classical_ising_symmetric_3D(ising_βc_3D)

"""
$(SIGNATURES)

Constructs the partition function tensor for a 3D cubic lattice
for the classical Ising model with a given inverse temperature `β` and coupling constant `J` (by default J = `1.0`).
    
### Examples
```julia
    classical_ising_3D() # Default inverse temperature is `ising_βc_3D`, coupling constant is `J = 1.0`.
    classical_ising_3D(0.5; J = 1.0) # Custom inverse temperature and coupling constant.
```

See also: [`classical_ising_symmetric_3D`](@ref), [`classical_ising`](@ref), [`classical_ising_symmetric`](@ref).
"""
function classical_ising_3D(β; J = 1.0)
    K = β * J

    # Boltzmann weights
    t = ComplexF64[exp(K) exp(-K); exp(-K) exp(K)]
    r = eigen(t)
    q = r.vectors * sqrt(LinearAlgebra.Diagonal(r.values)) * r.vectors

    # local partition function tensor
    O = zeros(2, 2, 2, 2, 2, 2)
    O[1, 1, 1, 1, 1, 1] = 1
    O[2, 2, 2, 2, 2, 2] = 1
    @tensor o[-1 -2; -3 -4 -5 -6] := O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] *
        q[-4; 4] * q[-5; 5] * q[-6; 6]

    TMS = ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)'

    return TensorMap(o, TMS)
end
classical_ising_3D() = classical_ising_3D(ising_βc_3D)
