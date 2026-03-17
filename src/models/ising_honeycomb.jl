const ising_βc_honeycomb = BigFloat(BigFloat(asinh(BigFloat(sqrt(BigFloat(3.0))))) / BigFloat(2.0))
const f_onsager_honeycomb::BigFloat = -1.556707467816387475214957698255679494804

"""
$(SIGNATURES)

Constructs the partition function tensor for a 2D honeycomb lattice
for the classical Ising model with a given inverse temperature `β`.

### Examples
```julia
    classical_ising_honeycomb() # Default inverse temperature is `ising_βc_honeycomb`
    classical_ising_honeycomb(0.5; h = 1.0) # Custom inverse temperature.
```

See also: [`classical_ising_honeycomb_symmetric`](@ref).
"""
function classical_ising_honeycomb(β; T = Float64)
    t = T[exp(β) exp(-β); exp(-β) exp(β)]

    r = eigen(t)
    nt = r.vectors * sqrt(LinearAlgebra.Diagonal(r.values)) * r.vectors

    O = zeros(2, 2, 2)
    O[1, 1, 1] = 1
    O[2, 2, 2] = 1

    H = [1 1; 1 -1] / sqrt(2)

    @tensor o[-1 -2 -3] := O[1 2 3] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3]
    @tensor o2[-1 -2 -3] := o[1 2 3] * H[-1; 1] * H[-2; 2] * H[-3; 3]

    return TensorMap(o2, ℂ^2 * ℂ^2, ℂ^2)
end
classical_ising_honeycomb() = classical_ising_honeycomb(ising_βc_honeycomb)

"""
$(SIGNATURES)

Constructs the partition function tensor for a symmetric 2D honeycomb lattice
for the classical Ising model with a given inverse temperature `β`.

This tensor has explicit ℤ₂ symmetry on each of it spaces.

### Examples
```julia
    classical_ising_honeycomb_symmetric() # Default inverse temperature is `ising_βc_honeycomb`
    classical_ising_honeycomb_symmetric(0.5) # Custom inverse temperature.
```

See also: [`classical_ising_honeycomb`](@ref).
"""
function classical_ising_honeycomb_symmetric(β; T = Float64)
    x = cosh(β)
    y = sinh(β)

    S = ℤ₂Space(0 => 1, 1 => 1)
    tens = zeros(T, S ⊗ S ← S)

    block(tens, Irrep[ℤ₂](0)) .= [2 * x^(3 / 2); 2 * sqrt(x) * y;;]
    block(tens, Irrep[ℤ₂](1)) .= [2 * sqrt(x) * y; 2 * sqrt(x) * y;;]
    return tens
end
classical_ising_honeycomb_symmetric() = classical_ising_honeycomb_symmetric(ising_βc_honeycomb)
