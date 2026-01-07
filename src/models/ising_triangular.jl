const ising_βc_triangular = log(big(3))/4
const f_onsager_triangular::BigFloat = -3.20253248660790791834355252025862951439

"""
$(SIGNATURES)

Constructs the partition function tensor for a 2D triangular lattice
for the classical Ising model with a given inverse temperature `β`.

### Examples
```julia
    classical_ising_triangular() # Default inverse temperature is `ising_βc_triangular`
    classical_ising_triangular(0.5; h = 1.0) # Custom inverse temperature.
```

See also: [`classical_ising_triangular_symmetric`](@ref).
"""
function classical_ising_triangular(β)
    t = Float64[exp(β) exp(-β); exp(-β) exp(β)]

    r = eigen(t)
    nt = r.vectors * sqrt(LinearAlgebra.Diagonal(r.values)) * r.vectors

    O = zeros(2, 2, 2, 2, 2, 2)
    O[1, 1, 1, 1, 1, 1] = 1
    O[2, 2, 2, 2, 2, 2] = 1

    H = [1 1; 1 -1] / sqrt(2)

    @tensor o[-1 -2 -3; -4 -5 -6] := O[1 2 3; 4 5 6] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * nt[-4; 4] * nt[-5; 5] * nt[-6; 6]
    @tensor o2[-1 -2 -3; -4 -5 -6] := o[1 2 3; 4 5 6] * H[-1; 1] * H[-2; 2] * H[-3; 3] * H[-4; 4] * H[-5; 5] * H[-6; 6]

    return TensorMap(o2, ℂ^2 * ℂ^2 * ℂ^2, ℂ^2 * ℂ^2 * ℂ^2)
end
classical_ising_triangular() = classical_ising_triangular(ising_βc_triangular)

"""
$(SIGNATURES)

Constructs the partition function tensor for a symmetric 2D triangular lattice
for the classical Ising model with a given inverse temperature `β`.

This tensor has explicit ℤ₂ symmetry on each of it spaces.

### Examples
```julia
    classical_ising_triangular_symmetric() # Default inverse temperature is `ising_βc_triangular`
    classical_ising_triangular_symmetric(0.5) # Custom inverse temperature.
```

See also: [`classical_ising_triangular`](@ref).
"""
function classical_ising_triangular_symmetric(β)
    x = cosh(β)
    y = sinh(β)

    S = ℤ₂Space(0 => 1, 1 => 1)
    T = zeros(Float64, S ⊗ S ⊗ S ← S ⊗ S ⊗ S)

    block(T, Irrep[ℤ₂](0)) .= [2 * x^3 2 * x^2 * y 2 * x^2 * y 2 * x^2 * y; 2 * x^2 * y 2 * x * y^2 2 * x * y^2 2 * x * y^2; 2 * x^2 * y 2 * x * y^2 2 * x * y^2 2 * x * y^2; 2 * x^2 * y 2 * x * y^2 2 * x * y^2 2 * x * y^2]
    block(T, Irrep[ℤ₂](1)) .= [2 * x^2 * y 2 * x^2 * y 2 * x^2 * y 2 * x * y^2; 2 * x^2 * y 2 * x^2 * y 2 * x^2 * y 2 * x * y^2; 2 * x^2 * y 2 * x^2 * y 2 * x^2 * y 2 * x * y^2; 2 * x * y^2 2 * x * y^2 2 * x * y^2 2 * y^3]
    return T
end
classical_ising_triangular_symmetric() = classical_ising_triangular_symmetric(ising_βc_triangular)
