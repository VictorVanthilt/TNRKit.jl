"""
$(SIGNATURES)

Constructs the partition function tensor for a 2D triangular lattice on the dual graph
for the classical Ising model with a given inverse temperature `β`.

### Examples
```julia
    classical_ising_dual_triangular() # Default inverse temperature is `ising_βc_triangular`
```

See also: [`classical_ising_dual_triangular`](@ref).
"""
function classical_ising_dual_triangular(β) # Ref: 10.1103/physrevlett.99.120601
    T_ele = zeros(Float64, 2, 2, 2)
    α = Float64(exp(β / 2))
    T_ele[1, 1, 1] = α^3
    T_ele[1, 2, 2] = 1 / α
    T_ele[2, 1, 2] = 1 / α
    T_ele[2, 2, 1] = 1 / α

    return TensorMap(T_ele, ℝ^2 ← ℝ^2 ⊗ ℝ^2)
end
classical_ising_dual_triangular() = classical_ising_dual_triangular(ising_βc_triangular)

"""
$(SIGNATURES)

Constructs the partition function tensor for a 2D triangular lattice on the dual graph
for the classical Ising model with a given inverse temperature `β`.

This tensor has explicit ℤ₂ symmetry on each of it spaces.

### Examples
```julia
    classical_ising_dual_triangular_symmetric() # Default inverse temperature is `ising_βc_triangular`
    classical_ising_dual_triangular_symmetric(0.5) # Custom inverse temperature.
```

See also: [`classical_ising_dual_triangular_symmetric`](@ref).
"""
function classical_ising_dual_triangular_symmetric(β) # Ref: 10.1103/physrevlett.99.120601
    V = Z2Space(0 => 1, 1 => 1)
    α = Float64(exp(β / 2))
    T = ones(Float64, V ← V ⊗ V) / α
    T[(0, 0, 0)] .= α^3
    return T
end
classical_ising_dual_triangular_symmetric() = classical_ising_dual_triangular_symmetric(ising_βc_triangular)

#     ↙     ↘     ↙     ↘     ↙
#  T           T           T
#  ↓           ↓           ↓
#  T'          T'          T'
#     ↘     ↙     ↘     ↙     ↘
#        T           T
#        ↓           ↓
#        T'          T'
#     ↙     ↘     ↙     ↘     ↙
#  T           T           T
#  ↓           ↓           ↓
#  T'          T'          T'
#     ↘     ↙     ↘     ↙     ↘

# contraction ⟶

#        |  ↘     ↙  |
#        ↓     A     ↓
#        |  ↙     ↘  |
#        C           B
#     ↙  |           |  ↘     ↙
#  A     ↓     ↺     ↓     A
#     ↘  |           |  ↙     ↘
#        B           C
#        |  ↘     ↙  |
#        ↓     A     ↓
#        |  ↙     ↘  |
#        C           B

# where
# A:
#  |     |       ↘     ↙
#  ↘     ↙          T
#     A        =    ↓
#  ↙     ↘          T'
#  |     |       ↙     ↘
#
# B:
#  |    |        |          ↓
#    ↘  ↓        |          T'
#       B      =   ↘     ↙     ↘
#       ↓  ↘          T          |
#       |    |        ↓          |
#
# C:
#       |    |         ↓          |
#       ↓  ↙           T'         |
#       C      =    ↙     ↘     ↙
#    ↓  ↓         |          T
#  |    |         |          ↓

function honeycomb_to_kagome(T::AbstractTensorMap{E, S, 1, 2}) where {E, S}
    TA = T' * T
    @tensor TB[-1 -2; -3 -4] := T'[5 -2; -4] * T[-1; -3 5]
    @tensor TC[-1 -2; -3 -4] := T[-2; 5 -4] * T'[-1 5; -3]
    return TA, TB, TC
end
