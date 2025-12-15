const ising_βc_honeycomb::BigFloat = log(3) / 4
# const f_onsager_triangular::BigFloat = 

"""
$(SIGNATURES)

Constructs the partition function tensor for a 2D honeycomb lattice
for the classical Ising model with a given inverse temperature `β`.

### Examples
```julia
    classical_ising_honeycomb() # Default inverse temperature is `ising_βc_honeycomb`
```

See also: [`classical_ising_triangular_symmetric`](@ref).
"""
function classical_ising_honeycomb(β)
    T_ele = zeros(Float64, 2, 2, 2)
    α = exp(-2 * β)
    T_ele[1, 1, 1] = 1.0
    T_ele[1, 2, 2] = α
    T_ele[2, 1, 2] = α
    T_ele[2, 2, 1] = α

    return TensorMap(T_ele, ℝ^2 ← ℝ^2 ⊗ ℝ^2)
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
function classical_ising_honeycomb_symmetric(β)
    V = Z2Space(0 => 1, 1 => 1)
    α = Float64(exp(-2 * β))
    T = ones(Float64, V ← V ⊗ V) * α
    T[(0, 0, 0)] .= 1.0
    return T
end
classical_ising_honeycomb_symmetric() = classical_ising_honeycomb_symmetric(ising_βc_honeycomb)

function honeycomb_to_kagome(T::AbstractTensorMap{E, S, 1, 2}) where {E, S}
    TA = T' * T
    @tensor TB[-1 -2; -3 -4] := T'[5 -2; -4] * T[-1; -3 5]
    @tensor TC[-1 -2; -3 -4] := T[-2; 5 -4] * T'[-1 5; -3]
    return TA, TB, TC
end