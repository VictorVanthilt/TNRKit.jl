const ising_βc_triangular = BigFloat(BigFloat(asinh(BigFloat(sqrt(BigFloat(1.0) / BigFloat(3.0))))) / BigFloat(2.0))
const f_onsager_triangular::BigFloat = -3.20253248660790791834355252025862951439

"""
    classical_ising_triangular(::Type{Trivial}, β::Real; T::Type{<:Number} = Float64)
    classical_ising_triangular(::Type{Z2Irrep}, β::Real; T::Type{<:Number} = Float64)

Constructs the partition function tensor for a 2D triangular lattice
for the classical Ising model with a given inverse temperature `β`.
Compatible with no symmetry or with explicit ℤ₂ symmetry on each of its spaces.
Defaults to ℤ₂ symmetry and inverse temperature `ising_βc_triangular` if the symmetry type and inverse temperature are not provided.

### Examples
```julia
    classical_ising_triangular() # Default ℤ₂ symmetry, inverse temperature is `ising_βc_triangular`
    classical_ising_triangular(Trivial, 0.5) # Custom inverse temperature wihout symmetry.
    classical_ising_triangular(0.5) # Custom inverse temperature with ℤ₂ symmetry.
```
"""
function classical_ising_triangular(β::Real; kwargs...)
    return classical_ising_triangular(Z2Irrep, β; kwargs...)
end
classical_ising_triangular(; kwargs...) = classical_ising_triangular(ising_βc_triangular; kwargs...)
classical_ising_triangular(::Type{Trivial}; kwargs...) = classical_ising_triangular(Trivial, ising_βc_triangular; kwargs...)
function classical_ising_triangular(::Type{Trivial}, β::Real; T::Type{<:Number} = Float64)
    t = T[exp(β) exp(-β); exp(-β) exp(β)]

    r = eigen(t)
    nt = r.vectors * sqrt(LinearAlgebra.Diagonal(r.values)) * r.vectors

    O = zeros(T, 2, 2, 2, 2, 2, 2)
    O[1, 1, 1, 1, 1, 1] = one(T)
    O[2, 2, 2, 2, 2, 2] = one(T)

    H = T[1 1; 1 -1] / sqrt(2)

    @tensor o[-1 -2 -3; -4 -5 -6] := O[1 2 3; 4 5 6] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * nt[-4; 4] * nt[-5; 5] * nt[-6; 6]
    @tensor o2[-1 -2 -3; -4 -5 -6] := o[1 2 3; 4 5 6] * H[-1; 1] * H[-2; 2] * H[-3; 3] * H[-4; 4] * H[-5; 5] * H[-6; 6]

    return TensorMap(o2, ℂ^2 * ℂ^2 * ℂ^2, ℂ^2 * ℂ^2 * ℂ^2)
end
function classical_ising_triangular(::Type{Z2Irrep}, β::Real; T::Type{<:Number} = Float64)
    x = cosh(β)
    y = sinh(β)

    S = ℤ₂Space(0 => 1, 1 => 1)
    t = zeros(T, S ⊗ S ⊗ S ← S ⊗ S ⊗ S)

    A = 2 * x^2 * y
    B = 2 * x * y^2
    block(t, Irrep[ℤ₂](0)) .= [2 * x^3 A A A; A B B B; A B B B; A B B B]
    block(t, Irrep[ℤ₂](1)) .= [A A A B; A A A B; A A A B; B B B 2 * y^3]
    return t
end
