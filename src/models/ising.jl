const ising_βc = BigFloat(log(BigFloat(1.0) + sqrt(BigFloat(2.0))) / BigFloat(2.0))
const f_onsager::BigFloat = -2.10965114460820745966777928351108478082549327543540531781696107967700291143188081390114126499095041781
const ising_cft_exact = [
    1 / 8, 1, 9 / 8, 9 / 8, 2, 2, 2, 2, 17 / 8, 17 / 8, 17 / 8, 3, 3,
    3, 3, 3,
    25 / 8, 25 / 8, 25 / 8, 25 / 8, 25 / 8, 25 / 8,
]
const ising_βc_3D = 1.0 / 4.51152469

function ising_bond_tensor(β::Real, T::Type{<:Number})
    x = cosh(β)
    y = sinh(β)
    bond_matrix = T[sqrt(x) 0; 0 sqrt(y)]
    return TensorMap(bond_matrix, ℂ^2 ← ℂ^2)
end

"""
    classical_ising(; kwargs...)
    classical_ising(β::Real; kwargs...)
    classical_ising(::Type{Trivial}, β::Real; T::Type{<:Number} = Float64, h = 0.0)
    classical_ising(::Type{Z2Irrep}, β::Real; T::Type{<:Number} = Float64, h = 0.0)

Constructs the partition function tensor for a 2D square lattice
for the classical Ising model with a given inverse temperature `β` and external magnetic field `h`.
Compatible with no symmetry for `h ≠ 0` or with explicit ℤ₂ symmetry for `h = 0` on each of its spaces.
Defaults to ℤ₂ symmetry and `h = 0` if the symmetry type and magnetic field are not provided.

### Examples
```julia
    classical_ising() # Default symmetry is `Z2Irrep`, default inverse temperature is `ising_βc` and default magnetic field `h = 0`.
    classical_ising(Trivial, 0.5; h = 1.0) # Custom inverse temperature without symmetry and custom magnetic field `h`.

!!! info
    When studying this model with impurities, the tensor without symmetry should be constructed, as the impurity breaks the ℤ₂ symmetry.
```

See also: [`classical_ising_3D`](@ref).
"""
function classical_ising(β::Real; kwargs...)
    return classical_ising(Z2Irrep, β; kwargs...)
end
classical_ising(; kwargs...) = classical_ising(ising_βc; kwargs...)
classical_ising(::Type{Trivial}; kwargs...) = classical_ising(Trivial, ising_βc; kwargs...)
function classical_ising(::Type{Trivial}, β::Real; T::Type{<:Number} = Float64, h = 0.0)
    init = zeros(T, 2, 2, 2, 2)
    for (i, j, k, l) in Iterators.product([1:2 for _ in 1:4]...)
        init[i, j, k, l] = mod(i + j + k + l, 2) == 0 ? cosh(h * β) : sinh(h * β)
    end
    init = TensorMap(init, ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2)

    bond_tensor = ising_bond_tensor(β, T)

    @tensor T[-1 -2; -3 -4] := 2 * init[1 2; 3 4] * bond_tensor[-1; 1] * bond_tensor[-2; 2] * bond_tensor[3; -3] * bond_tensor[4; -4]
    return T
end
function classical_ising(::Type{Z2Irrep}, β::Real; T::Type{<:Number} = Float64, h = 0.0)
    @assert h == 0.0 "External magnetic field is not compatible with ℤ₂ symmetry"
    x = cosh(β)
    y = sinh(β)

    S = ℤ₂Space(0 => 1, 1 => 1)
    t = zeros(T, S ⊗ S ← S ⊗ S)
    block(t, Irrep[ℤ₂](0)) .= [2x^2 2x * y; 2x * y 2y^2]
    block(t, Irrep[ℤ₂](1)) .= [2x * y 2x * y; 2x * y 2x * y]

    return t
end

"""
    classical_ising_impurity([Type{Trivial}], β::Real; T::Type{<:Number} = Float64, h = 0.0)

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

See also: [`classical_ising`](@ref), [`classical_ising_3D`](@ref).
"""
function classical_ising_impurity(β::Real; kwargs...)
    return classical_ising_impurity(Trivial, β; kwargs...)
end
classical_ising_impurity(; kwargs...) = classical_ising_impurity(ising_βc; kwargs...)
function classical_ising_impurity(::Type{Trivial}, β::Real; T::Type{<:Number} = Float64, h = 0.0)
    init = zeros(T, 2, 2, 2, 2)
    for (i, j, k, l) in Iterators.product([1:2 for _ in 1:4]...)
        init[i, j, k, l] = mod(i + j + k + l, 2) == 0 ? sinh(h * β) : cosh(h * β)
    end
    init = TensorMap(init, ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2)

    bond_tensor = ising_bond_tensor(β, T)

    @tensor t[-1 -2; -3 -4] := 2 * init[1 2; 3 4] * bond_tensor[-1; 1] * bond_tensor[-2; 2] * bond_tensor[3; -3] * bond_tensor[4; -4]
    return t
end

"""
    classical_ising_3D(; kwargs...)
    classical_ising_3D(β::Real; kwargs...)
    classical_ising_3D(::Type{Trivial}, β::Real; T::Type{<:Number} = Float64, J = 1.0)
    classical_ising_3D(::Type{Z2Irrep}, β::Real; T::Type{<:Number} = Float64, J = 1.0)

Constructs the partition function tensor for a symmetric 3D cubic lattice
for the classical Ising model with a given inverse temperature `β`.

Compatible with no symmetry or with explicit ℤ₂ symmetry on each of its spaces.
Defaults to ℤ₂ symmetry and coupling constant `J = 1.0` if the symmetry type and coupling constant are not provided.

### Examples
```julia
    classical_ising_3D() # Default ℤ₂ symmetry, inverse temperature is `ising_βc_3D`, coupling constant is `J = 1.0`.
    classical_ising_3D(Trivial, 0.5; J = 1.5) # Custom inverse temperature and coupling constant.
    classical_ising_3D(Z2Irrep, 0.5; J = 1.5) # Custom inverse temperature and coupling constant with ℤ₂ symmetry.
```

See also: [`classical_ising`](@ref).
"""
function classical_ising_3D(β::Real; kwargs...)
    return classical_ising_3D(Z2Irrep, β; kwargs...)
end
classical_ising_3D(; kwargs...) = classical_ising_3D(ising_βc_3D; kwargs...)
classical_ising_3D(::Type{Trivial}; kwargs...) = classical_ising_3D(Trivial, ising_βc_3D; kwargs...)
function classical_ising_3D(::Type{Trivial}, β::Real; T::Type{<:Number} = Float64, J = 1.0)
    K = β * J

    # Boltzmann weights
    t = T[exp(K) exp(-K); exp(-K) exp(K)]
    r = eigen(t)
    q = r.vectors * sqrt(LinearAlgebra.Diagonal(r.values)) * r.vectors

    # local partition function tensor
    O = zeros(T, 2, 2, 2, 2, 2, 2)
    O[1, 1, 1, 1, 1, 1] = 1
    O[2, 2, 2, 2, 2, 2] = 1
    @tensor o[-1 -2; -3 -4 -5 -6] := O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] *
        q[-4; 4] * q[-5; 5] * q[-6; 6]

    TMS = ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)'

    return TensorMap(o, TMS)
end
function classical_ising_3D(::Type{Z2Irrep}, β::Real; T::Type{<:Number} = Float64, J = 1.0)
    x = cosh(β * J)
    y = sinh(β * J)
    W = T[sqrt(x) sqrt(y); sqrt(x) -sqrt(y)]
    t_array = zeros(T, 2, 2, 2, 2, 2, 2)
    for (i, j, k, l, m, n) in Iterators.product([1:2 for _ in 1:6]...)
        for a in 1:2
            # Outer product of W[a, :] with itself 6 times
            t_array[i, j, k, l, m, n] += W[a, i] * W[a, j] * W[a, k] * W[a, l] * W[a, m] *
                W[a, n]
        end
    end
    S = ℤ₂Space(0 => 1, 1 => 1)
    t = TensorMap(t_array, S ⊗ S ⊗ S ← S ⊗ S ⊗ S)

    return permute(t, ((1, 4), (5, 6, 2, 3)))
end
