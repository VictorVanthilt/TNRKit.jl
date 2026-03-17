const ising_βc = BigFloat(log(BigFloat(1.0) + sqrt(BigFloat(2.0))) / BigFloat(2.0))
const f_onsager::BigFloat = -2.10965114460820745966777928351108478082549327543540531781696107967700291143188081390114126499095041781
const ising_cft_exact = [
    1 / 8, 1, 9 / 8, 9 / 8, 2, 2, 2, 2, 17 / 8, 17 / 8, 17 / 8, 3, 3,
    3, 3, 3,
    25 / 8, 25 / 8, 25 / 8, 25 / 8, 25 / 8, 25 / 8,
]
const ising_βc_3D = 1.0 / 4.51152469

function ising_bond_tensor(β::Real)
    elt = bigfloat_convert(β; warn = false)
    x = cosh(β)
    y = sinh(β)
    bond_matrix = elt[sqrt(x) 0; 0 sqrt(y)]
    return TensorMap(bond_matrix, ℂ^2 ← ℂ^2)
end

function bigfloat_convert(β::Real; warn = true)
    isbigfloat = β isa BigFloat
    elt = isbigfloat ? Float64 : typeof(β)
    isbigfloat && warn && @warn "β is a BigFloat, but the tensor will be constructed with Float64 precision"
    return elt
end

"""
    classical_ising(::Type{Trivial}, β::Real; h = 0.0)
    classical_ising(::Type{Z2Irrep}, β::Real; h = 0.0)

Constructs the partition function tensor for a 2D square lattice
for the classical Ising model with a given inverse temperature `β` and external magnetic field `h`.
Compatible with no symmetry for `h ≠ 0` or with explicit ℤ₂ symmetry for `h = 0` on each of its spaces.
Defaults to ℤ₂ symmetry and `h = 0` if the symmetry type and magnetic field are not provided.

### Examples
```julia
    classical_ising() # Default symmetry is `Z2Irrep`, default inverse temperature is `ising_βc` and default magnetic field `h = 0`.
    classical_ising(Trivial, 0.5; h = 1.0) # Custom inverse temperature with explicit ℤ₂ symmetry with custom magnetic field `h`.
```

See also: [`classical_ising_3D`](@ref).
"""
function classical_ising(β::Real; h = 0.0)
    return classical_ising(Z2Irrep, β; h = h)
end
classical_ising() = classical_ising(ising_βc)
classical_ising(::Type{Trivial}) = classical_ising(Trivial, ising_βc)
function classical_ising(::Type{Trivial}, β::Real; h = 0.0)
    elt = bigfloat_convert(β)
    init = zeros(elt, 2, 2, 2, 2)
    for (i, j, k, l) in Iterators.product([1:2 for _ in 1:4]...)
        init[i, j, k, l] = mod(i + j + k + l, 2) == 0 ? cosh(h * β) : sinh(h * β)
    end
    init = TensorMap(init, ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2)

    bond_tensor = ising_bond_tensor(elt(β))

    @tensor T[-1 -2; -3 -4] := 2 * init[1 2; 3 4] * bond_tensor[-1; 1] * bond_tensor[-2; 2] * bond_tensor[3; -3] * bond_tensor[4; -4]
    return T
end
function classical_ising(::Type{Z2Irrep}, β::Real; h = 0.0)
    elt = bigfloat_convert(β)
    @assert h == 0.0 "External magnetic field is not compatible with ℤ₂ symmetry"
    x = cosh(β)
    y = sinh(β)

    S = ℤ₂Space(0 => 1, 1 => 1)
    T = zeros(elt, S ⊗ S ← S ⊗ S)
    block(T, Irrep[ℤ₂](0)) .= [2x^2 2x * y; 2x * y 2y^2]
    block(T, Irrep[ℤ₂](1)) .= [2x * y 2x * y; 2x * y 2x * y]

    return T
end

"""
    classical_ising_impurity([Type{Trivial}], β::Real; h = 0.0)

Constructs the partition function tensor for a 2D square lattice
for the classical Ising model with a given inverse temperature `β` and external magnetic field `h` with a magnetisation impurity.
Compatible with no symmetry on each of its spaces.

### Examples
```julia
    classical_ising_impurity() # Default inverse temperature is `ising_βc`
    classical_ising_impurity(0.5; h = 1.0) # Custom inverse temperature and magnetic field
```
!!! info
    When calculating the free energy with `free_energy()`, set the `initial_size` keyword argument to `2.0`.
    The initial lattice holds 2 spins.

See also: [`classical_ising_3D`](@ref).
"""
function classical_ising_impurity(β::Real; h = 0.0)
    return classical_ising_impurity(Trivial, β; h = h)
end
classical_ising_impurity() = classical_ising_impurity(ising_βc)
function classical_ising_impurity(::Type{Trivial}, β::Real; h = 0.0)
    elt = bigfloat_convert(β)

    init = zeros(elt, 2, 2, 2, 2)
    for (i, j, k, l) in Iterators.product([1:2 for _ in 1:4]...)
        init[i, j, k, l] = mod(i + j + k + l, 2) == 0 ? sinh(h * β) : cosh(h * β)
    end
    init = TensorMap(init, ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2)

    bond_tensor = ising_bond_tensor(β)

    @tensor T[-1 -2; -3 -4] := 2 * init[1 2; 3 4] * bond_tensor[-1; 1] * bond_tensor[-2; 2] * bond_tensor[3; -3] * bond_tensor[4; -4]
    return T
end

"""
    classical_ising_3D(::Type{Trivial}, β::Real; J = 1.0)
    classical_ising_3D(::Type{Z2Irrep}, β::Real; J = 1.0)

Constructs the partition function tensor for a symmetric 3D cubic lattice
for the classical Ising model with a given inverse temperature `β`.

Compatible with no symmetry or with explicit ℤ₂ symmetry on each of its spaces.
Defaults to ℤ₂ symmetry and coupling constant `J = 1.0` if the symmetry type and coupling constant are not provided.

### Examples
```julia
    classical_ising_3D() # Default ℤ₂ symmetry, inverse temperature is `ising_βc_3D`, coupling constant is `J = 1.0`.
    classical_ising_3D(Trivial, 0.5; J = 1.0) # Custom inverse temperature and coupling constant.
    classical_ising_3D(Z2Irrep, 0.5; J = 1.5) # Custom inverse temperature and coupling constant with ℤ₂ symmetry.
```

See also: [`classical_ising`](@ref).
"""
function classical_ising_3D(β::Real; J = 1.0)
    return classical_ising_3D(Z2Irrep, β; J = J)
end
classical_ising_3D() = classical_ising_3D(ising_βc_3D)
classical_ising_3D(::Type{Trivial}) = classical_ising_3D(Trivial, ising_βc_3D)
function classical_ising_3D(::Type{Trivial}, β::Real; J = 1.0)
    elt = bigfloat_convert(β)
    K = β * J

    # Boltzmann weights
    t = elt[exp(K) exp(-K); exp(-K) exp(K)]
    r = eigen(t)
    q = r.vectors * sqrt(LinearAlgebra.Diagonal(r.values)) * r.vectors

    # local partition function tensor
    O = zeros(elt, 2, 2, 2, 2, 2, 2)
    O[1, 1, 1, 1, 1, 1] = 1
    O[2, 2, 2, 2, 2, 2] = 1
    @tensor o[-1 -2; -3 -4 -5 -6] := O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] *
        q[-4; 4] * q[-5; 5] * q[-6; 6]

    TMS = ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)'

    return TensorMap(o, TMS)
end
function classical_ising_3D(::Type{Z2Irrep}, β::Real; J = 1.0)
    @assert J == 1.0 "Coupling constant must be one for ℤ₂ symmetry" # FIXME: is this true?
    elt = bigfloat_convert(β)
    x = cosh(β)
    y = sinh(β)
    W = [sqrt(x) sqrt(y); sqrt(x) -sqrt(y)]
    T_array = zeros(elt, 2, 2, 2, 2, 2, 2)
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
